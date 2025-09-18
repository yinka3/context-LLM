from datetime import datetime, timedelta
import logging
from entity_model import ERTransformer
import logging_setup
import json
import os
from redisclient import RedisClient
from redis import Redis
from typing import Any, Dict, List, Tuple
import networkx as nx
from networkx import DiGraph
from nlp_pipe import NLP_PIPE
from dtypes import EntityData, MessageData
from vectordb import ChromaClient
from rq import Queue
from fact_check import FactExtractor

logging_setup.setup_logging()

logger = logging.getLogger(__name__)

class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.next_id = 1
        self.user_name = user_name
        self.graph: DiGraph = DiGraph()
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.history: List[MessageData] = []
        self.entities: Dict[int, EntityData] = {}
        self.chroma: ChromaClient = ChromaClient()
        self.redis_client = RedisClient()
        self.rq_client = Redis(password=os.getenv('REDIS_PASSWORD'))
        self.user_message_cnt: int = 0
        self.bridge_map: Dict[str, Dict[Any, List[int]]] = {}
        self.queue: Queue = Queue(connection=self.rq_client)
        self.alias_index: Dict[str, EntityData] = {}
        
        self.user_entity = self._create_user_entity(user_name)

        self.ENTITY_RESOLVER = ERTransformer(
            graph=self.graph,
            chroma=self.chroma,
            user_entity=self.user_entity,
            next_id=self.get_new_id,
            redis_client=self.redis_client,
            alias_index=self.alias_index
        )
        self.FACT_EXTRACTOR = FactExtractor(entity_resolver=self.ENTITY_RESOLVER)

    def _create_user_entity(self, user_name: str) -> EntityData:
        user_entity = EntityData(
            id=0, 
            name="USER", 
            type="person",
            aliases=[{"text": user_name, "type": "person"}]
        )
        ent_id = f"ent_{user_entity.id}"
        self.graph.add_node(ent_id, data=user_entity, type="entity")
        self.entities[user_entity.id] = user_entity
        self.chroma.add_item(user_entity)
        return user_entity
    
    def get_new_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    
    def add(self, item: MessageData):
        self.history.append(item)
        if item.role == "user":
            self.user_message_cnt += 1
        self.process_live_messages(item)

    
    def get_recent_context(self, num_messages: int = 10) -> Tuple[str, List[str]]:
        # Get from Redis cache
        sorted_set_key = f"recent_messages:{self.user_name}"
        recent_msg_ids = self.redis_client.client.zrevrange(sorted_set_key, 0, num_messages-1)
        
        context_text = []
        for msg_id in recent_msg_ids:
            msg_data = self.redis_client.client.hget(f"message_content:{self.user_name}", msg_id)
            if msg_data:
                context_text.append(json.loads(msg_data)['message'])
        
        return " ".join(context_text), context_text


    def search_graph(self, src_node: str, depth: int = 2):

        subgraph = nx.bfs_tree(G=self.graph, source=src_node, depth_limit=depth)
        context_node_ids = sorted(subgraph.nodes())
        return context_node_ids
    
    def track_active_entities(self, resolved_entities: Dict, timestamp: datetime):
        """Keep track of recently mentioned entities"""
        active_key = f"active_entities:{self.user_name}"
        
        for entity in resolved_entities.values():
            # Store with timestamp as score for automatic expiry
            self.redis_client.client.zadd(
                active_key,
                {f"ent_{entity.id}": timestamp.timestamp()}
            )
        
        # Keep only last hour's entities
        cutoff = (timestamp - timedelta(hours=1)).timestamp()
        self.redis_client.client.zremrangebyscore(active_key, 0, cutoff)

    def add_to_redis(self, msg: MessageData):
        sorted_set_key = f"recent_messages:{self.user_name}" 
        hash_key = f"message_content:{self.user_name}"

        self.redis_client.client.hset(hash_key, f"msg_{msg.id}", json.dumps({
            'message': msg.message,
            'timestamp': msg.timestamp.isoformat(),
            'role': msg.role
        }))

        self.redis_client.client.zadd(sorted_set_key, {f"msg_{msg.id}": msg.timestamp.timestamp()})
        self.redis_client.client.zremrangebyrank(sorted_set_key, 0, -76)
    
    def entity_context(self, resolved_entities: Dict):
        all_context_ids = set()
        for entity in resolved_entities.values():
            ent_id = f"ent_{entity.id}"
            if self.graph.has_node(ent_id):
                related_ids = self.search_graph(src_node=ent_id)
                all_context_ids.update(related_ids)
        
        return all_context_ids

    def process_live_messages(self, msg: MessageData):
        
        msg_id = f"msg_{msg.id}"
        self.graph.add_node(msg_id, data=msg, type="message")
        self.add_to_redis(msg)
        

        results = self.nlp_pipe.start_process(message_block=self.get_recent_context(), 
                                              msg=msg, entity_threshold=0.7)

        if results.get("tier2_flags") != []:
            for flag in results["tier2_flags"]:
                logger.warning(f"Received Tier 2 flag from NLP_PIPE. Reason: {flag['reason']}")
            return
        
        if not results.get("high_confidence_entities"):
            logger.info("No entities found")
            return
        
        recent_context = self.get_recent_context()
        resolved_entities = self.ENTITY_RESOLVER.resolve_entities_with_coreference(
            entities_from_nlp=results["high_confidence_entities"],
            coref_clusters=results.get("coref_clusters", []),
            msg_id=msg_id,
            conversational_context=recent_context,
            appositive_map=results.get("appositive_map")
        )

        self.track_active_entities(resolved_entities, msg.timestamp)


        for entity in resolved_entities.values():
            self.alias_index[entity.name.lower()] = entity
            for alias in entity.aliases:
                self.alias_index[alias["text"].lower()] = entity

        















