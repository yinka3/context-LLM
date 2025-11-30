import asyncio
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import logging
import logging_setup
import json
import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI
from redis import exceptions
from redisclient import AsyncRedisClient
from typing import TYPE_CHECKING, Dict, List, Tuple, TypeVar, Type
from functools import partial
from schema.dtypes import *

from schema.common_pb2 import Entity, Relationship, BatchMessage
if TYPE_CHECKING:
    from main.nlp_pipe import NLPPipeline
    from main.entity_resolve import EntityResolver
    from graph.memgraph import MemGraphStore

logging_setup.setup_logging()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
STREAM_KEY_AI_RESPONSE = "stream:ai_response"
LLM_CLIENT = lambda: instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
)

def _log_task_exception(task):
    if exc := task.exception():
        logger.error(f"Background task failed: {exc}")

class Context:
    def __init__(self, user_name: str, topics: List[str], redis_client, llm_client):
        self.user_name: str = user_name
        self.active_topics: List[str] = topics
        
        self.redis_client: redis.Redis = redis_client 
        self.llm_client: AsyncOpenAI = llm_client
        
        self.store: 'MemGraphStore' = None 
        self.cpu_executor: ThreadPoolExecutor = None
        self.nlp_pipe: 'NLPPipeline' = None
        self.ent_resolver: 'EntityResolver' = None

    @classmethod
    async def create(cls, user_name: str = "Yinka", topics: List[str] = ["General"]) -> "Context":
        redis_conn = AsyncRedisClient().get_client()
        llm = LLM_CLIENT()
        
        instance = cls(user_name, topics, redis_conn, llm)
        
        instance.store = MemGraphStore()
        instance.cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ctx_worker")
        
        loop = asyncio.get_running_loop()
        
        instance.nlp_pipe = await loop.run_in_executor(
            instance.cpu_executor, NLPPipeline
        )
        
        instance.ent_resolver = await loop.run_in_executor(
            instance.cpu_executor, EntityResolver
        )

        initialized = await loop.run_in_executor(
            instance.cpu_executor, 
            instance.ent_resolver._init_from_db 
        )

        if not initialized:
            logger.critical("Entity resolver is not hydrated, STOP PROGRAM")
            raise RuntimeError("Failed to initialize EntityResolver from DB")

        await instance._get_or_create_user_entity(user_name)
        
        return instance


    async def get_next_msg_id(self) -> int:
        return await self.redis_client.incr("global:next_msg_id")

    async def get_next_ent_id(self) -> int:
        return await self.redis_client.incr("global:next_ent_id")


    async def _get_or_create_user_entity(self, user_name):

        profile = self.ent_resolver.entity_profiles.get(
            self.ent_resolver.fuzzy_choices.get(user_name))

        if profile:
            logger.info(f"User {user_name} recognized.")
            return profile
        
        logger.info(f"Creating new USER entity for {user_name}")
        new_id = await self.get_next_ent_id()

        profile = {
            "canonical_name": "USER",
            "aliases": [user_name],
            "summary": f"The primary user named {user_name}",
            "type": "PERSON"
        }

        embedding_vector = await asyncio.get_running_loop().run_in_executor(
            self.cpu_executor, 
            partial(self.ent_resolver.add_entity, new_id, profile)
        )

        user_entity = Entity(
            id=new_id,
            text="USER",
            type="PERSON",
            confidence=1.0,
            aliases=[],
            summary=profile["summary"],
            topic="Meta",
            embedding=embedding_vector
        )

        batch = BatchMessage(message_id=0)
        batch.list_ents.append(user_entity)

        try:
            await self.redis_client.xadd(STREAM_KEY_AI_RESPONSE, {'data': batch.SerializeToString()})
        except Exception as e:
            logger.error(f"Failed to push User Entity to stream: {e}")
        
    
    async def add(self, msg: MessageData):
        msg.id = await self.get_next_msg_id()
        await self.add_to_redis(msg)
        task = asyncio.create_task(self.process_live_messages(msg))
        task.add_done_callback(_log_task_exception)

    
    async def get_recent_context(self, num_messages: int = 10) -> List[str]:
        sorted_set_key = f"recent_messages:{self.user_name}"
        recent_msg_ids = await self.redis_client.zrevrange(sorted_set_key, 0, num_messages-1)
        
        if not recent_msg_ids:
            return []
        
        recent_msg_ids.reverse()
        
        msg_data_list = await self.redis_client.hmget(
            f"message_content:{self.user_name}", 
            *recent_msg_ids
        )
        
        context_text = []
        for msg_data in msg_data_list:
            if msg_data:
                context_text.append(json.loads(msg_data)['message'])
    
        return context_text


    async def add_to_redis(self, msg: MessageData):
        msg_key = f"msg_{msg.id}"
        
        pipe = self.redis_client.pipeline()

        pipe.hset(f"message_content:{self.user_name}", msg_key, json.dumps({
            'message': msg.message.strip(),
            'timestamp': msg.timestamp.isoformat()
        }))
        pipe.zadd(f"recent_messages:{self.user_name}", {msg_key: msg.timestamp.timestamp()})
        pipe.zremrangebyrank(f"recent_messages:{self.user_name}", 0, -76)
        await pipe.execute()
    
    def _build_extraction_prompt(
        self, 
        msg: MessageData, 
        resolution_candidates: Dict[str, List[Dict]],
        recent_history: List[str]
    ) -> Tuple[str, str]:
    
        topics_str = ", ".join(self.active_topics) if self.active_topics else "General"

        system_prompt = f"""You are a Knowledge Graph extraction engine for a personal memory system.

            The user talks about their life - people they know, places they go, projects they work on.
            Your job is to extract entities and relationships from their messages.

            **AVAILABLE TOPICS**: {topics_str}

            **TASK 1: ENTITY EXTRACTION & RESOLUTION**
            - Extract entities (people, organizations, places, projects, events, etc.)
            - Check if each entity matches a provided candidate. If yes, use the candidate's `id`. If new, use `null`.
            - Assign each entity to one of the available topics. If none fit, use "General".
            - Set `has_new_info: true` if the message reveals NEW facts about an existing entity.

            **TASK 2: RELATIONSHIP EXTRACTION**
            - Identify factual connections between entities.
            - Use canonical names for source and target.

            **EXAMPLES**

            Input: "Jake started working at Stripe last week"
            Candidates: {{"Jake": [{{"id": 7, "name": "Jake", "summary": "User's friend from college"}}]}}
            Output:
            {{
            "entities": [
                {{"id": 7, "canonical_name": "Jake", "type": "PERSON", "topic": "Friends", "confidence": 0.95, "has_new_info": true}},
                {{"id": null, "canonical_name": "Stripe", "type": "ORG", "topic": "Work", "confidence": 0.9, "has_new_info": false}}
            ],
            "relationships": [
                {{"source": "Jake", "target": "Stripe", "relation": "works_at", "confidence": 0.95}}
            ]
            }}

            Input: "Had coffee with my sister"
            Candidates: {{"sister": [{{"id": 3, "name": "Sarah", "summary": "User's older sister, lives in Boston"}}]}}
            Output:
            {{
            "entities": [
                {{"id": 3, "canonical_name": "Sarah", "type": "PERSON", "topic": "Family", "confidence": 0.9, "has_new_info": false}}
            ],
            "relationships": []
            }}
            """

        clean_candidates = {}
        for mention, c_list in resolution_candidates.items():
            clean_candidates[mention] = [
                {"id": c["id"], 
                    "name": c.get("profile", {}).get("canonical_name", "Unknown"), 
                    "summary": c.get("profile", {}).get("summary", "No information provided yet.")}
                for c in c_list
            ]

        user_prompt_data = {
            "conversation_history": recent_history,
            "current_message": msg.message.strip(),
            "candidates": clean_candidates
        }
        
        return system_prompt, json.dumps(user_prompt_data, indent=2)
    
    def _build_profiling_prompt(
        self, 
        entity_name: str, 
        entity_type: str,
        existing_profile: Dict, 
        new_observation: str,
        conversation_context: List[str]
    ) -> Tuple[str, str]:
    
        system_prompt = """You are a Profile Refinement engine for a personal memory system.

        **TASK:**
        1. Review the existing profile and new observation.
        2. Update the summary ONLY if there are genuinely new facts.
        3. Preserve existing facts unless contradicted.
        4. Keep summaries concise (2-3 sentences max).
        5. Consolidate any new aliases.

        **EXAMPLE**

        Existing profile:
        {"canonical_name": "Jake", "aliases": [], "summary": "User's friend from college", "topic": "Personal"}

        New observation: "Jake started his new job at Stripe as a backend engineer"

        Updated profile:
        {"canonical_name": "Jake", "aliases": [], "summary": "User's friend from college. Works at Stripe as a backend engineer.", "topic": "Personal"}
        """

        user_prompt_data = {
            "entity_target": entity_name,
            "entity_type": entity_type,
            "existing_profile": existing_profile,
            "new_observation": new_observation,
            "recent_context": conversation_context
        }

        return system_prompt, json.dumps(user_prompt_data, indent=2)

    
    async def _call_slm(self, 
                        prompt: Tuple[str, str], 
                        response_model: Type[T]) -> T | None:
        
        system, user = prompt
        
        try:
            return await self.llm_client.chat.completions.create(
                model="qwen2.5:14b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_model=response_model,
                max_retries=2
            )
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return None
    
    async def _send_batch_to_stream(self, message_id: int, entities: List[Entity], relations: List[Relationship]):
        """Helper to serialize and push to Redis Stream"""
        batch = BatchMessage(
            message_id=message_id,
            list_ents=entities,
            list_relations=relations
        )
        try:
            await self.redis_client.xadd(STREAM_KEY_AI_RESPONSE, {'data': batch.SerializeToString()})
        except exceptions.RedisError as e:
            logger.error(f"Failed to push batch to stream: {e}")
    
    async def _run_profile_updates(self, 
        entities_list: List[Dict], 
        context_text: str,
        recent_context_list: List[str]):
        
        """Background task to run Prompt 2 for specific entities"""

        async def update_single_entity(ent):
            ent_id = ent["id"]
            existing_profile = self.ent_resolver.entity_profiles.get(ent_id, {})
            old_summary = existing_profile.get("summary", "No information provided")
            
            prompt_profile = self._build_profiling_prompt(
                entity_name=ent["canonical_name"],
                entity_type=ent["type"],
                existing_profile=existing_profile,
                new_observation=context_text,
                conversation_context=recent_context_list
            )
            updated_profile = await self._call_slm(prompt_profile, response_model=ProfileUpdate)

            if not updated_profile:
                return None
            
            current_aliases = set(existing_profile.get("aliases", []))
            current_aliases.update(updated_profile.aliases)
            
            old_name = existing_profile.get("canonical_name")
            if old_name and old_name != updated_profile.canonical_name:
                current_aliases.add(old_name)
            
            if updated_profile.canonical_name in current_aliases:
                current_aliases.remove(updated_profile.canonical_name)
                
            final_aliases_list = list(current_aliases)
            new_summary = updated_profile.summary
        
            if new_summary == old_summary:
                return None

            profile_dict = {
                "canonical_name": updated_profile.canonical_name,
                "aliases": final_aliases_list,
                "summary": new_summary,
                "topic": updated_profile.topic,
            }
            
            new_embedding = await asyncio.get_running_loop().run_in_executor(
                self.cpu_executor,
                partial(self.ent_resolver.update_entity, ent_id, profile_dict)
            )
                
            return Entity(
                id=ent_id,
                text=updated_profile.canonical_name,
                type=ent["type"],
                aliases=updated_profile.aliases,
                summary=new_summary,
                topic=updated_profile.topic,
                embedding=new_embedding
            )
        
        tasks = [update_single_entity(ent) for ent in entities_list]
        results = await asyncio.gather(*tasks)
        updates_for_graph = [r for r in results if r is not None]
        
        if updates_for_graph:
            await self._send_batch_to_stream(
                message_id=0,
                entities=updates_for_graph,
                relations=[]
            )
            logger.info(f"Refined {len(updates_for_graph)} profiles in background.")


    async def process_live_messages(self, msg: MessageData):
        
        recent_context_list = await self.get_recent_context()

        loop = asyncio.get_running_loop()

        mentions = await loop.run_in_executor(
            self.cpu_executor,
            self.nlp_pipe.extract_mentions,
            msg.message.strip(),
            0.5)

        resolution_candidates = {}
        tasks = [
            loop.run_in_executor(
                self.cpu_executor,
                partial(self.ent_resolver.resolve, text=mention, context=msg.message.strip(), top_k=3))
            for mention in mentions
        ]
        results = await asyncio.gather(*tasks)

        resolution_candidates = {
            mention: candidates 
            for mention, candidates in zip(mentions, results) 
            if candidates}

        prompt_extract = self._build_extraction_prompt(
            msg, resolution_candidates, recent_context_list)
        
        extraction_response = await self._call_slm(prompt_extract, ExtractionResponse)
        
        if not extraction_response:
            return
        
        final_entities: List[Dict] = []
        entities_needing_profile: List[Dict] = []

        new_entities = []
        existing_entities = []

        for ent in extraction_response.entities:
            if ent.id is None:
                new_entities.append(ent)
            else:
                existing_entities.append(ent)

        tasks = []
        id_map = {}

        for ent in new_entities:
            final_id = await self.get_next_ent_id()
            id_map[id(ent)] = final_id
            
            stub_profile = {
                "canonical_name": ent.canonical_name,
                "aliases": [],
                "summary": f"No information on {ent.canonical_name}",
                "type": ent.type
            }
            
            tasks.append(
                loop.run_in_executor(
                    self.cpu_executor,
                    partial(self.ent_resolver.add_entity, final_id, stub_profile)
                )
            )
            
        embedding_results = await asyncio.gather(*tasks) if tasks else []

        for i, ent in enumerate(new_entities):
            final_id = id_map[id(ent)]
            entity_dict = {
                "id": final_id,
                "canonical_name": ent.canonical_name,
                "topic": ent.topic,
                "type": ent.type,
                "aliases": [],
                "confidence": ent.confidence,
                "embedding": embedding_results[i],
                "has_new_info": ent.has_new_info,
            }
            final_entities.append(entity_dict)
            entities_needing_profile.append(entity_dict)

        for ent in existing_entities:
            existing_profile = self.ent_resolver.entity_profiles.get(ent.id, {})
            entity_dict = {
                "id": ent.id,
                "canonical_name": ent.canonical_name,
                "topic": ent.topic,
                "type": ent.type,
                "aliases": existing_profile.get("aliases", []),
                "confidence": ent.confidence,
                "embedding": None,
                "has_new_info": ent.has_new_info,
            }
            final_entities.append(entity_dict)
            if ent.has_new_info:
                entities_needing_profile.append(entity_dict)
        
        relationships = [
            {
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation,
                "confidence": rel.confidence
            }
            for rel in extraction_response.relationships
        ]
        
        await self.publish_structure(final_entities, relationships, msg.id)

        if entities_needing_profile:
            asyncio.create_task(
                self._run_profile_updates(entities_needing_profile, msg.message.strip(), recent_context_list)
            )
        
    
    async def publish_structure(self, normalized_entities: List[Dict], relationships: List[Dict], msg_id: int):
        """
        Sends the Association Graph structure to Memgraph.
        """
        proto_ents = []
        for ent in normalized_entities:
            new_ent = Entity(
                id=ent["id"],
                text=ent["canonical_name"],
                type=ent["type"],
                confidence=ent.get("confidence", 1.0),
                topic=ent.get("topic", "General"),
                embedding=ent["embedding"]
            )
            proto_ents.append(new_ent)
        
        proto_rels = []
        for rel in relationships:
            new_rel = Relationship(
                source_text=rel["source"],
                target_text=rel["target"],
                confidence=rel.get("confidence", 1.0),
                relation=rel.get("relation", "related")
            )
            proto_rels.append(new_rel)
        
        await self._send_batch_to_stream(msg_id, proto_ents, proto_rels)
        logger.info(f"Published structure for msg_{msg_id}: {len(proto_ents)} ents, {len(proto_rels)} rels")
    
    async def shutdown(self):
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.redis_client:
            await self.redis_client.aclose()
    



