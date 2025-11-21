import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import logging_setup
import json
from main.entity_resolve import EntityResolver
from redisclient import RedisClient
from redis import exceptions
from typing import Dict, List, Tuple
from main.nlp_pipe import NLP_PIPE
from shared.dtypes import MessageData
from models.factory import get_llm_client
from schema.common_pb2 import Entity, Relationship, BatchMessage

logging_setup.setup_logging()

logger = logging.getLogger(__name__)
STREAM_KEY_AI_RESPONSE = "stream:ai_response"


class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.redis_client = RedisClient()
        self.cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ctx_worker")
        

        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        
        self.ent_resolver: EntityResolver = EntityResolver()
        self.ent_resolver.load()
        
        try:
            self.llm_client = get_llm_client()
        except ValueError as e:
            logger.critical(f"Failed to initialize LLM: {e}")
            raise

        self.user_name = user_name
        
        self.user_entity = self._get_or_create_user_entity(user_name)


    def get_next_msg_id(self) -> int:
        return self.redis_client.client.incr("global:next_msg_id")

    def get_next_ent_id(self) -> int:
        return self.redis_client.client.incr("global:next_ent_id")


    def _get_or_create_user_entity(self, user_name):

        if user_name in self.ent_resolver.fuzzy_choices:
            existing_id = self.ent_resolver.fuzzy_choices[user_name]
            logger.info(f"User {user_name} recognized with ID {existing_id}")
            return self.ent_resolver.entity_profiles.get(existing_id)
        
        logger.info("Adding USER information to graph")
        new_id = self.get_next_ent_id()
        user_entity = Entity(
            id=new_id,
            text="USER",
            type="PERSON",
            confidence=1.0,
            aliases=[user_name],
            mentioned_in=[]
        )

        seriliazed_message = user_entity.SerializeToString()
        self.redis_client.client.xadd("stream-direct:add_user", {'data': seriliazed_message})

        profile = {
            "canonical_name": "USER",
            "aliases": [user_name],
            "summary": f"The primary user named {user_name}",
            "type": "PERSON"
        }
        self.ent_resolver.add_entity(new_id, profile)
        self.ent_resolver.save()
        return user_entity
        
    
    def add(self, msg: MessageData):
        asyncio.create_task(self.process_live_messages(msg))

    
    def get_recent_context(self, num_messages: int = 10) -> Tuple[str, List[str]]:
        sorted_set_key = f"recent_messages:{self.user_name}"
        recent_msg_ids = self.redis_client.client.zrevrange(sorted_set_key, 0, num_messages-1)
        
        recent_msg_ids.reverse()

        context_text = []
        for msg_id in recent_msg_ids:
            msg_data = self.redis_client.client.hget(f"message_content:{self.user_name}", msg_id)
            if msg_data:
                context_text.append(json.loads(msg_data)['message'])
        
        return " ".join(context_text), context_text


    def add_to_redis(self, msg: MessageData):
        msg_key = f"msg_{msg.id}"
        
        self.redis_client.client.hset(f"message_content:{self.user_name}", msg_key, json.dumps({
            'message': msg.message,
            'timestamp': msg.timestamp,
            'role': msg.role
        }))

        self.redis_client.client.zadd(f"recent_messages:{self.user_name}", {msg_key: float(msg.timestamp)})
        self.redis_client.client.zremrangebyrank(f"recent_messages:{self.user_name}", 0, -76)
    
    def _build_extraction_prompt(self, msg: MessageData, nlp_results: Dict, 
                                    resolution_candidates: Dict[str, List[Dict]],
                                    recent_history: List[str]) -> Tuple[str, str]:
        """
        PROMPT 1: The Graph Architect.
        Focus: ID Resolution (Link vs Create), Relationship Extraction, and Significance Filtering.
        """
        
        system_prompt = """You are a precise Knowledge Graph extraction engine. Your goal is to map entities and extract relationships without hallucinating details.

    **TASK 1: ENTITY RESOLUTION**
    - I will provide `potential_entities` found in the text and `candidates` from the database.
    - **LINKING:** If a potential entity refers to a candidate, you MUST use that candidate's `id` (Integer).
    - **CREATING:** If it is completely new, use `null` for `id`.
    - **SIGNIFICANCE:** Set `has_new_info`: true ONLY if the text provides specific, factual updates about the entity (e.g., "John moved to NY"). Set `false` for casual mentions (e.g., "Hi John").

    **TASK 2: RELATIONSHIP EXTRACTION**
    - Extract factual relationships between entities.
    - Use the resolved `canonical_name` for source and target.

    **OUTPUT JSON SCHEMA:**
    {
    "entities": [
        {
        "id": int | null,
        "canonical_name": "string",
        "type": "string", 
        "has_new_info": boolean
        }
    ],
    "relationships": [
        {
        "source": "string",
        "target": "string",
        "relation": "string",
        "confidence": float
        }
    ]
    }"""
        
        clean_candidates = {}
        for mention, c_list in resolution_candidates.items():
            clean_candidates[mention] = [
                {"id": c["id"], "name": c["profile"]["canonical_name"], "summary": c["profile"]["summary"]}
                for c in c_list
            ]

        user_prompt_data = {
            "conversation_history": recent_history[-5:], # Keep history short for extraction
            "current_message": msg.message,
            "nlp_analysis": self.clean_nlp_for_prompt(nlp_results),
            "candidates": clean_candidates
        }
        
        return system_prompt, json.dumps(user_prompt_data, indent=2)
    
    def _build_profiling_prompt(self, entity_name: str, existing_profile: Dict, 
                                new_observation: str) -> Tuple[str, str]:
        """
        PROMPT 2: The Biographer.
        Focus: Synthesis, Summary Writing, and Alias Consolidation.
        Only called if 'has_new_info' was True in the extraction step.
        """
        
        system_prompt = """You are a Profile Refinement engine for a digital memory system.
        
**TASK:**
1. Read the `existing_profile` and the `new_observation`.
2. Update the `summary` to incorporate the new facts naturally.
3. **CONSTRAINT:** Keep the summary concise (under 50 words). Focus on identity, role, and key facts.
4. Consolidate `aliases` (add new ones, remove duplicates).
5. Refine the `type` if the new info is more specific.

**OUTPUT JSON SCHEMA:**
{
  "canonical_name": "string",
  "type": "string",
  "aliases": ["string"],
  "summary": "string"
}"""

        user_prompt_data = {
            "entity_target": entity_name,
            "existing_profile": existing_profile,
            "new_observation": new_observation
        }

        return system_prompt, json.dumps(user_prompt_data, indent=2)

    
    async def _call_llm(self, prompt_tuple: Tuple[str, str]) -> Dict | None:
        system, user = prompt_tuple
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.cpu_executor,
            lambda: self.llm_client.generate_json_response(system, user)
        )

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

    async def _update_resolver_from_llm(self, response):
        
        if "resolved_entities" not in response:
            logger.warning("LLM response missing 'resolved_entities'. Cannot update resolver")
            return

        for entity in response["resolved_entities"]:
            ent_id = entity.get("id")

            profile = {
                "canonical_name": entity["canonical_name"],
                "aliases": entity.get("aliases", []),
                "summary": entity.get("summary", ""),
                "type": entity.get("entity_type", "MISC")
            }

            if ent_id:
                self.ent_resolver.update_entity(int(ent_id), profile)
            else:
                new_ent_id = self.get_next_ent_id()
                entity["id"] = new_ent_id
                logger.info(f"Adding new entity {new_ent_id} to resolver.")
                self.ent_resolver.add_entity(
                    entity_id=new_ent_id,
                    profile=profile
                )
        
        logger.info("Saving Entity Resolver state to disk...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.cpu_executor, self.ent_resolver.save)
    
    async def _run_profile_updates(self, entities_list: List[Dict], context_text: str):
        """Background task to run Prompt 2 for specific entities"""
        for ent in entities_list:
            ent_id = ent.get("id")
            name = ent.get("canonical_name")
            
            existing_profile = {}
            if ent_id:
                existing_profile = self.ent_resolver.entity_profiles.get(int(ent_id), {})
            
            prompt_profile = self._build_profiling_prompt(name, existing_profile, context_text)
            updated_profile = await self._call_llm(prompt_profile)

            if updated_profile:
                final_id = int(ent_id) if ent_id else self.get_next_ent_id()
                
                self.ent_resolver.update_entity(final_id, updated_profile)
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.cpu_executor, self.ent_resolver.save)


    async def process_live_messages(self, msg: MessageData):
        
        self.add_to_redis(msg)
        recent_context = self.get_recent_context()

        loop = asyncio.get_running_loop()

        results = await loop.run_in_executor(
            self.cpu_executor,
            self.nlp_pipe.start_process,
            recent_context,
            msg,
            0.65
        )
        
        
        potential_mentions = results.get("high_confidence_entities", [])
        resolution_candidates = {}

        for mention in potential_mentions:
            mention_text = mention["text"]
            candidates = self.ent_resolver.resolve(text=mention_text, context=msg.message, top_k=3)
            if candidates:
                resolution_candidates[mention_text] = candidates

        prompt_extract = self._build_extraction_prompt(
            msg, results, resolution_candidates, recent_context[1]
        )
        extraction_response = await self._call_llm(prompt_extract)
        
        if not extraction_response:
            return
        
        self.publish_message(extraction_response, msg)

        updates_needed = []
        for ent in extraction_response.get("entities", []):
            if ent["id"] is None or ent.get("has_new_info", False):
                updates_needed.append(ent)
        
        if updates_needed:
            asyncio.create_task(
                self._run_profile_updates(updates_needed, msg.message)
            )
        
    
    def publish_message(self, llm_response, msg=MessageData):
        
        batched_msg = BatchMessage(message_id=msg.id)
        for ent in llm_response["resolved_entities"]:

            raw_id = ent.get("id")
            if raw_id is not None:
                entity_id = int(raw_id)
            else:
                entity_id = self.get_next_ent_id()

            new_ent = Entity(
                id=entity_id,
                text=ent.get("canonical_name"), 
                type=ent.get("entity_type"),
                confidence=ent.get("confidence", 1.0),
                aliases=ent.get("aliases", []),
                mentioned_in=[str(msg.id)] 
            )
            
            batched_msg.list_ents.append(new_ent)
        
        for relationship in llm_response["relationships"]:

            new_relation = Relationship(
                source_text=relationship["source"],
                target_text=relationship["target"],
                relation=relationship["relation"],
                confidence=relationship["confidence"]
            )
            
            batched_msg.list_relations.append(new_relation)
        
        serialized_data = batched_msg.SerializeToString()

        try:
            self.redis_client.client.xadd(STREAM_KEY_AI_RESPONSE, {'data': serialized_data})
            logger.info(f"Added message for {msg.id} to stream '{STREAM_KEY_AI_RESPONSE}'")
        except exceptions.RedisError as e:
            logger.error(f"Failed to add message to Redis Stream: {e}")








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


