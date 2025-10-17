import asyncio
from datetime import datetime, timedelta
import logging
import logging_setup
import json
import os
from redisclient import RedisClient
from redis import Redis
from typing import Any, Dict, List, Tuple
from main.nlp_pipe import NLP_PIPE
from shared.dtypes import EntityData, MessageData
from schema.common_pb2 import Entity, Relationship, BatchMessge, GraphResponse, Message

logging_setup.setup_logging()

logger = logging.getLogger(__name__)

class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.ents_id = 0
        self.msg_id = 1
        self.user_name = user_name
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.llm_client = None
        self.entities: Dict[int, EntityData] = {}
        self.redis_client = RedisClient()
        self.publish_id = 0
        self.user_message_cnt: int = 0
        self.bridge_map: Dict[str, Dict[Any, List[int]]] = {}
        self.alias_index: Dict[str, EntityData] = {}
        self.user_entity = self._create_user_entity(user_name)
    
    def get_next_msg_id(self) -> str:
        self.msg_id += 1
        return f"msg_{self.msg_id}"

    def get_next_ent_id(self) -> str:
        self.ents_id += 1
        return f"ent_{self.ents_id}"

    # maybe do this in graph builder instead of here
    def _create_user_entity(self):
        logger.info("Adding basic USER information to graph")
        user_entity = Entity(
            id=self.get_next_ent_id(),
            text="USER",
            type="PERSON",
            confidence=1.0,
            aliases=[{"text": self.user_name, "type": "PERSON"}],
            recieving_ents=[],
            mentioned_in=[]
        )

        seriliazed_message = user_entity.SerializeToString()
        try:
            self.redis_client.client.xadd("stream-direct:add_user", {'data': seriliazed_message})
            self.redis_client.client.xack()
        self.entities[user_entity.id] = user_entity
        return user_entity
        
    
    def add(self, item: Message):
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
    
    
    def _build_llm_prompt(self, msg: MessageData, nlp_results: Dict, recent_context: List[str]) -> Tuple[str, str]:
        """
        Builds the system and user prompts for the SLM.
        """
        system_prompt = f"""You are a semantic extraction engine for business knowledge graphs. Extract entities and relationships with precision and appropriate confidence scoring.

        ENTITY EXTRACTION:
        - VALIDATE: Confirm high_confidence_entities, correct types if needed
        - SCRUTINIZE: Accept/reject low_confidence_entities based on context
        - DISCOVER: Find missed entities using coref_clusters and conversation context

        ENTITY TYPES:
        Use these primary types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, WORK_PRODUCT_OR_PROJECT, TECHNOLOGY, EVENT, TOPIC, ACADEMIC_CONCEPT, POSSESSIVE_ENTITY, GROUP_OF_ENTITIES

        TOPIC ASSIGNMENT:
        User's active topics: {', '.join(self.active_topics)}
        
        For each entity, assign to the most relevant topic from the active list. If no active topic fits well, suggest a new topic name that better captures the entity's context.
        
        RELATIONSHIP EXTRACTION:
        Extract semantic relationships between entities using your reasoning capabilities. Consider these common patterns as examples, but discover ANY meaningful relationships present:

        EXAMPLE PATTERNS (not exhaustive):
        - Actions: "Sarah teaches Mike" → [teaches], "team plays against rivals" → [competes_with]
        - Possession: "Sarah's guitar" → [owns], "Mike's theory" → [authored/developed]  
        - Roles: "Sarah, the guitarist" → [has_role], "Mike, my brother" → [sibling_of]
        - Location: "meeting in library" → [located_in], "lives in Boston" → [resides_in]
        - Temporal: "concert tomorrow" → [scheduled_for], "after graduation" → [follows]
        - Social: "friends with", "dating", "mentored by" → [friends_with], [dating], [mentored_by]
        - Academic: "studies under", "researches", "cites" → [studies_under], [researches], [cites]
        - Creative: "inspired by", "covers song by", "collaborated on" → [inspired_by], [covers], [collaborated_on]

        DISCOVERY APPROACH:
        Look beyond these examples. Extract relationships from any domain - personal, academic, creative, technical, social, familial, professional, hobby-related, etc. Use your reasoning to identify meaningful connections between entities.

        RELATIONSHIP EXTRACTION METHODS:
        - Verbs connecting entities: "Sarah teaches guitar" → [teaches]
        - Possessives: "Mike's research" → [conducts/owns] 
        - Appositives: "Jake, my roommate" → [roommate_of]
        - Prepositions: "concert at venue" → [performed_at]
        - Contextual implications: "study group met" → [participates_in]
        - Implicit relationships: "They've been together 5 years" → [in_relationship_with]

        RELATIONSHIP NAMING:
        Use clear, descriptive relation names. Prefer specific verbs ("teaches", "lives_with", "studies") over generic ones ("relates_to", "associated_with").

        CONFIDENCE SCORING:
        ENTITIES: 1.0 (explicit/unambiguous), 0.8 (contextually clear), 0.6 (reasonably inferred), 0.4 (uncertain)
        RELATIONSHIPS: 1.0 (explicit verbs), 0.8 (clear syntax), 0.6 (contextual inference), 0.4 (weak signal)
        Reject entities/relationships below 0.4 confidence.

        GRAPH OPERATION SUGGESTIONS:
        Based on extraction results, suggest these operations:

        REQUIRED OPERATIONS (always suggest when applicable):
        - "add_entity": For each entity in resolved_entities 
        - "add_relationship": For each relationship extracted

        
        OUTPUT (JSON only):
        {
        "resolved_entities": [
            {
                "text": str,
                "type": str,
                "confidence": float
            }
        ],
        "relationships": [
            {
                "source": str,
                "target": str,
                "relation": str, 
                "confidence": float,
                "directionality": "bidirectional|source_to_target|target_to_source"
            }
        ],
        "topic_suggestions": [
            {
                "suggested_topic": str, 
                "entities": [list of entity texts]
            }
        ]
        }"""

        user_prompt_data = {
            "conversation_history": recent_context[-10:],
            "current_message": msg.message,
            "pre_analysis": self.clean_nlp_for_prompt(nlp_results)
        }
        
        return system_prompt, json.dumps(user_prompt_data, indent=2)

    
    def _call_llm(self, prompt):

        sys_prompt, user_prompt = prompt
        # call model
        # get response
        # parse to dict 
        # return dict
        pass

    def clean_nlp_for_prompt(self, result: Dict):
        analysis_payload = {}

        if result.get("high_confidence_entities"):
            analysis_payload["high_confidence_entities"] = [
                {"text": ent["text"], "type": ent["type"], "confidence": ent["confidence"], "contextual_mention": ent["contextual_mention"]}
                for ent in result["high_confidence_entities"]
            ]

        if result.get("low_confidence_entities"):
            analysis_payload["low_confidence_entities"] = [
                {"text": ent["text"], "type": ent["type"], "confidence": ent["confidence"], "contextual_mention": ent["contextual_mention"]}
                for ent in result["low_confidence_entities"]
            ]
        
        if result.get("time_expressions"):
            analysis_payload["time_expressions"] = result["time_expressions"]


        if result.get("emotion"):
            analysis_payload["emotion"] = result["emotion"]

        if result.get("coref_clusters"):
            analysis_payload["coref_clusters"] = [
                {
                    "main": cluster.main.text,
                    "mentions": [mention.text for mention in cluster.mentions]
                }
                for cluster in result["coref_clusters"]
            ]
        
        return analysis_payload



    def process_live_messages(self, msg: Message):
        
        self.add_to_redis(msg)

        #NOTE this should be a blocking operation right???
        results = self.nlp_pipe.start_process(message_block=self.get_recent_context(), 
                                              msg=msg, entity_threshold=0.7)
        
        recent_context, _ = self.get_recent_context()

        prompt = self._build_llm_prompt(msg, results, recent_context)
        llm_response = self._call_llm(prompt)
        self.publish_message(llm_response=llm_response, msg=msg)
        
    
    def publish_message(self, llm_response, msg=Message):
        
        batched_msg = BatchMessge(message_id=msg.id)
        for ent in llm_response["resolved_entities"]:

            new_ent = Entity(
                            id=self.get_next_ent_id(),
                            text=ent["text"], 
                            type=ent["type"],
                            topic_groups=ent["topic"],
                            confidence=ent["confidence"],
                            aliases=ent["aliases"],
                            mentioned_in=[msg.id]
                        )
            
            batched_msg.list_ents.append(new_ent)
        
        for relationship in llm_response["relationships"]:

            new_relation = Relationship(source_text=relationship["source_text"], 
                                        target_text=relationship["target_text"],
                                        relation=relationship["relation"],
                                        confidence=relationship["confidence"],
                                        directionality=relationship["directionality"])
            
            batched_msg.list_relations.append(new_relation)
        
        serialized_data = batched_msg.SerializeToString()
        self.redis_client.client.publish(channel="parser:ai_response", message=serialized_data)


if '__main__' == __name__:
    manager = Context()
    while True:
        manager.publish_message()
        asyncio.sleep(0.5)














