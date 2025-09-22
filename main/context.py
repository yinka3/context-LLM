from datetime import datetime, timedelta
import logging
import logging_setup
import json
import os
from main.redisclient import RedisClient
from redis import Redis
from typing import Any, Dict, List, Tuple
from main.nlp_pipe import NLP_PIPE
from shared.dtypes import EntityData, MessageData


logging_setup.setup_logging()

logger = logging.getLogger(__name__)

class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.next_id = 1
        self.user_name = user_name
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.llm_client = None
        self.entities: Dict[int, EntityData] = {}
        self.redis_client = RedisClient()
        self.rq_client = Redis(password=os.getenv('REDIS_PASSWORD'))
        self.user_message_cnt: int = 0
        self.bridge_map: Dict[str, Dict[Any, List[int]]] = {}
        self.alias_index: Dict[str, EntityData] = {}
        self.user_entity = self._create_user_entity(user_name)


    def _create_user_entity(self, user_name: str) -> EntityData:
        user_entity = EntityData(
            id=0, 
            name="USER", 
            type="person",
            aliases=[{"text": user_name, "type": "person"}]
        )
        ent_id = f"ent_{user_entity.id}"
        #NOTE send to graph through redis
        self.entities[user_entity.id] = user_entity
        self.chroma.add_item(user_entity)
        return user_entity
    
    def get_new_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    
    def add(self, item: MessageData):
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
    
    
    def _build_llm_prompt():
        pass

    def _call_llm():
        pass

    def _verify_response():
        pass

    def process_live_messages(self, msg: MessageData):
        
        msg_id = f"msg_{msg.id}"
        #NOTE send data to graph
        #self.graph.add_node(msg_id, data=msg, type="message")
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
        
        recent_context, _ = self.get_recent_context()

        prompt = self._build_llm_prompt(msg, results, recent_context)
        llm_response = self._call_llm(prompt)
        verify_response = self._verify_response(llm_response)
        #NOTE have to send all the information to graph
        

        















