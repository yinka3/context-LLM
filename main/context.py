import asyncio
from datetime import datetime, timedelta
import logging
import logging_setup
import json
import os
from main.entity_resolve import EntityResolver
from redisclient import RedisClient
from redis import exceptions
from typing import Any, Dict, List, Tuple
from main.nlp_pipe import NLP_PIPE
from shared.dtypes import EntityData, MessageData
from schema.common_pb2 import Entity, Relationship, BatchMessge, GraphResponse, Message

logging_setup.setup_logging()

logger = logging.getLogger(__name__)
STREAM_KEY_AI_RESPONSE = "stream:ai_response"
class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.ents_id = 0
        self.msg_id = 1
        self.user_message_cnt: int = 0
        self.user_name = user_name
        self.entities: Dict[int, EntityData] = {}
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.ent_resolver: EntityResolver = EntityResolver()
        self.redis_client = RedisClient()
        self.llm_client = None
        

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
    
    
    def _build_llm_prompt(self, msg: MessageData, nlp_results: Dict, resolution_candidates: Dict[str, List[Dict]],
                          recent_context: List[str]) -> Tuple[str, str]:
        """
        Builds the system and user prompts for the SLM.
        """
        system_prompt = f"""You are a semantic extraction and entity resolution engine. Your goal is to accurately identify entities in a user's message, linking them to existing known entities when possible, or identifying them as new.

        **ENTITY RESOLUTION TASK:**
        - For each potential entity found in the `pre_analysis`, I have provided a list of `entity_resolution_candidates`.
        - Review the `conversation_history` and the candidate `profile` summaries.
        - **DECISION:** For each entity, you MUST decide if it refers to one of the candidates OR if it is a completely new entity.
        - **LINKING:** If it matches a candidate, you MUST use that candidate's `id` in your output. The `text` in your output should be the candidate's canonical name. You may add newly discovered aliases.
        - **CREATING:** If it is a new entity, you MUST use `null` for the `id`.

        **EXTRACTION TASK:**
        - Extract semantic relationships between the final resolved entities.
        - Ensure relationship `source` and `target` fields use the final canonical names.
        
        **OUTPUT (JSON only):**
        {{
          "resolved_entities": [
            {{
              "id": str | null, // Use candidate's ID if matched, otherwise null
              "canonical_name": str, // The primary name for the entity
              "entity_type": str,
              "aliases": [str], // Include all known and new aliases
              "summary": str, // A brief, updated summary based on new context
              "confidence": float
            }}
          ],
          "relationships": [
            {{
              "source": str, // Canonical name of source entity
              "target": str, // Canonical name of target entity
              "relation": str, 
              "confidence": float
            }}
          ]
        }}"""

        user_prompt_data = {
            "conversation_history": recent_context[-10:],
            "current_message": msg.message,
            "pre_analysis": self.clean_nlp_for_prompt(nlp_results),
            "entity_resolution_candidates": resolution_candidates
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

    #TODO: make a profile data structure to keep consistency in context and entity resolver
    async def _update_resolver_from_llm(self, response):
        
        if "resolved_entities" not in response:
            logger.warning("LLM response missing 'resolved_entities'. Cannot update resolver")
            return

        for entity_profile in response["resolved_entities"]:
            ent_id = entity_profile.get("id")

            if ent_id:
                logger.info(f"Updating entity {ent_id} in resolver.")
                self.ent_resolver.update_entity(ent_id, entity_profile)
            else:
                new_ent_id = self.get_next_ent_id()
                entity_profile["id"] = new_ent_id
                logger.info(f"Adding new entity {new_ent_id} to resolver.")
                self.ent_resolver.add_entity(
                    entity_id=new_ent_id,
                    profile=entity_profile
                )
        logger.info("Entity resolver update task finished.")


    async def process_live_messages(self, msg: Message):
        
        self.add_to_redis(msg)
        recent_context = self.get_recent_context()

        #NOTE this should be a blocking operation right???
        results = self.nlp_pipe.start_process(message_block=recent_context, 
                                              msg=msg, entity_threshold=0.65)
        
        potential_mentions = results.get("high_confidence_entities", [])
        resolution_candidates = {}
        for mention in potential_mentions:
            mention_text = mention["text"]
            # Find top 3 candidates for this mention
            candidates = self.ent_resolver.resolve(text=mention_text, context=msg.message, top_k=3)
            if candidates:
                resolution_candidates[mention_text] = candidates

        prompt = self._build_llm_prompt(msg, results, resolution_candidates, recent_context[0])
        llm_response = self._call_llm(prompt)

        asyncio.create_task(self._update_resolver_from_llm(llm_response))
        self.publish_message(llm_response=llm_response, msg=msg)
        
    
    def publish_message(self, llm_response, msg=Message):
        
        batched_msg = BatchMessge(message_id=msg.id)
        for ent in llm_response["resolved_entities"]:
            entity_id = ent.get("id") or self.get_next_ent_id()
            new_ent = Entity(id=entity_id,
                            text=ent["text"], 
                            type=ent["entity_type"],
                            confidence=ent["confidence"],
                            aliases=ent.get("aliases", []),
                            mentioned_in=[msg.id]
                        )
            
            batched_msg.list_ents.append(new_ent)
        
        for relationship in llm_response["relationships"]:

            new_relation = Relationship(source_text=relationship["source_text"], 
                                        target_text=relationship["target_text"],
                                        relation=relationship["relation"],
                                        confidence=relationship["confidence"])
            
            batched_msg.list_relations.append(new_relation)
        
        serialized_data = batched_msg.SerializeToString()

        try:
            self.redis_client.client.xadd(STREAM_KEY_AI_RESPONSE, {'data': serialized_data})
            logger.info(f"Added message for {msg.id} to stream '{STREAM_KEY_AI_RESPONSE}'")
        except exceptions.RedisError as e:
            logger.error(f"Failed to add message to Redis Stream: {e}")


if '__main__' == __name__:
    manager = Context()
    while True:
        manager.publish_message()
        asyncio.sleep(0.5)











"""
def track_active_entities(self, resolved_entities: Dict, timestamp: datetime):
        Keep track of recently mentioned entities
        active_key = f"active_entities:{self.user_name}"
        
        for entity in resolved_entities.values():
            self.redis_client.client.zadd(
                active_key,
                {f"ent_{entity.id}": timestamp.timestamp()}
            )
        
        # Keep only last hour's entities
        cutoff = (timestamp - timedelta(hours=1)).timestamp()
        self.redis_client.client.zremrangebyscore(active_key, 0, cutoff)
"""


