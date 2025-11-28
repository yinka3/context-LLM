import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import logging_setup
import json
import instructor
from pydantic import BaseModel
from openai import OpenAI
from main.entity_resolve import EntityResolver
from redisclient import RedisClient
from redis import exceptions
from typing import Dict, List, Tuple, TypeVar, Type
from main.nlp_pipe import NLPPipeline
from schema.dtypes import *

from schema.common_pb2 import Entity, Relationship, BatchMessage
from graph.memgraph import MemGraphStore

logging_setup.setup_logging()

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
STREAM_KEY_AI_RESPONSE = "stream:ai_response"

LLM_CLIENT = lambda: instructor.patch(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))

class Context:

    def __init__(self, user_name: str = "Yinka", topics: List[str] = ["General"]):
        self.active_topics: List[str] = topics
        self.redis_client = RedisClient()
        self.store = MemGraphStore()
        self.cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ctx_worker")
        
        self.nlp_pipe = NLPPipeline()
        
        self.ent_resolver: EntityResolver = EntityResolver()
        self.ent_resolver.load()
        
        try:
            self.llm_client = LLM_CLIENT()
            self.llm_client.models.list()
        except Exception as e:
            logger.critical(f"Failed to initialize LLM: {e}")
            raise
        
        self.user_name = user_name
        self._get_or_create_user_entity(user_name)


    def get_next_msg_id(self) -> int:
        return self.redis_client.client.incr("global:next_msg_id")

    def get_next_ent_id(self) -> int:
        return self.redis_client.client.incr("global:next_ent_id")


    def _get_or_create_user_entity(self, user_name):

        if user_name in self.ent_resolver.fuzzy_choices:
            existing_id = self.ent_resolver.fuzzy_choices[user_name]
            logger.info(f"User {user_name} recognized with ID {existing_id}")
            return self.ent_resolver.entity_profiles.get(existing_id)
        
        logger.info(f"Creating new USER entity for {user_name}")
        new_id = self.get_next_ent_id()

        profile = {
            "canonical_name": "USER",
            "aliases": [user_name],
            "summary": f"The primary user named {user_name}",
            "type": "PERSON"
        }

        embedding_vector = self.ent_resolver.add_entity(new_id, profile)

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
            self.redis_client.client.xadd(STREAM_KEY_AI_RESPONSE, {'data': batch.SerializeToString()})
        except Exception as e:
            logger.error(f"Failed to push User Entity to stream: {e}")
        
    
    def add(self, msg: MessageData):
        asyncio.create_task(self.process_live_messages(msg))

    
    def get_recent_context(self, num_messages: int = 10) -> List[str]:
        sorted_set_key = f"recent_messages:{self.user_name}"
        recent_msg_ids = self.redis_client.client.zrevrange(sorted_set_key, 0, num_messages-1)
        
        recent_msg_ids.reverse()

        context_text = []
        for msg_id in recent_msg_ids:
            msg_data = self.redis_client.client.hget(f"message_content:{self.user_name}", msg_id)
            if msg_data:
                context_text.append(json.loads(msg_data)['message'])
        
        return context_text


    def add_to_redis(self, msg: MessageData):
        msg_key = f"msg_{msg.id}"
        
        self.redis_client.client.hset(f"message_content:{self.user_name}", msg_key, json.dumps({
            'message': msg.message.strip(),
            'timestamp': msg.timestamp.isoformat(),
            'role': msg.role
        }))

        self.redis_client.client.zadd(f"recent_messages:{self.user_name}", {msg_key: msg.timestamp.timestamp()})
        self.redis_client.client.zremrangebyrank(f"recent_messages:{self.user_name}", 0, -76)
    
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
                {"id": c["id"], "name": c["profile"]["canonical_name"], "summary": c["profile"].get("summary", "")}
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

    
    async def _call_llm(self, 
                        prompt: Tuple[str, str], 
                        response_model: Type[T]) -> T | None:
        
        system, user = prompt
        loop = asyncio.get_running_loop()
        
        try:
            return await loop.run_in_executor(
                self.cpu_executor,
                lambda: self.llm_client.chat.completions.create(
                    model="qwen2.5:14b",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    response_model=response_model,
                    max_retries=2
                )
            )
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return None
    
    async def _run_profile_updates(self, 
        entities_list: List[Dict], 
        context_text: str,
        recent_context_list: List[str]
    ):
        """Background task to run Prompt 2 for specific entities"""

        updates_for_graph = []

        for ent in entities_list:
            ent_id = ent["id"]
            name = ent["canonical_name"]
            
            existing_profile = self.ent_resolver.entity_profiles.get(ent_id, {})
            old_summary = existing_profile.get("summary", "No information provided")
            
            prompt_profile = self._build_profiling_prompt(
                entity_name=name,
                entity_type=ent["type"],
                existing_profile=existing_profile,
                new_observation=context_text,
                conversation_context=recent_context_list
            )
            updated_profile = await self._call_llm(prompt_profile, response_model=ProfileUpdate)

            if not updated_profile:
                continue
            
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
                continue

            profile_dict = {
                "canonical_name": updated_profile.canonical_name,
                "aliases": final_aliases_list,
                "summary": new_summary,
                "topic": updated_profile.topic,
            }
            
            new_embedding = self.ent_resolver.update_entity(ent_id, profile_dict)
                
            graph_update = Entity(
                id=ent_id,
                text=updated_profile.canonical_name,
                type=ent["type"],
                aliases=updated_profile.aliases,
                summary=new_summary,
                topic=updated_profile.topic,
                embedding=new_embedding
            )
            updates_for_graph.append(graph_update)
        
        
        if updates_for_graph:
            self._send_batch_to_stream(
                message_id=0,
                entities=updates_for_graph,
                relations=[]
            )
            logger.info(f"Refined {len(updates_for_graph)} profiles in background.")
    

    def _send_batch_to_stream(self, message_id: int, entities: List[Entity], relations: List[Relationship]):
        """Helper to serialize and push to Redis Stream"""
        batch = BatchMessage(
            message_id=message_id,
            list_ents=entities,
            list_relations=relations
        )
        try:
            self.redis_client.client.xadd(STREAM_KEY_AI_RESPONSE, {'data': batch.SerializeToString()})
        except exceptions.RedisError as e:
            logger.error(f"Failed to push batch to stream: {e}")


    async def process_live_messages(self, msg: MessageData):
        
        self.add_to_redis(msg)
        recent_context_list = self.get_recent_context()

        loop = asyncio.get_running_loop()

        mentions = await loop.run_in_executor(
        self.cpu_executor,
        self.nlp_pipe.extract_mentions,
        msg.message.strip(),
        0.5)

        resolution_candidates = {}
        for mention in mentions:
            candidates = self.ent_resolver.resolve(text=mention, context=msg.message.strip(), top_k=3)
            if candidates:
                resolution_candidates[mention] = candidates

        prompt_extract = self._build_extraction_prompt(
            msg, resolution_candidates, recent_context_list)
        
        extraction_response = await self._call_llm(prompt_extract, ExtractionResponse)
        
        if not extraction_response:
            return
        
        final_entities: List[Dict] = []
        entities_needing_profile: List[Dict] = []

        for ent in extraction_response.entities:
            is_new = ent.id is None
            
            if is_new:
                final_id = self.get_next_ent_id()
                logger.info(f"Generated NEW ID {final_id} for entity '{ent.canonical_name}'")

                stub_profile = {
                    "canonical_name": ent.canonical_name,
                    "aliases": [],
                    "summary": f"No information on {ent.canonical_name}",
                    "type": ent.type
                }
                embedding_vec = self.ent_resolver.add_entity(final_id, stub_profile)
            else:
                existing_profile = self.ent_resolver.entity_profiles.get(ent.id, {})
                current_aliases = existing_profile.get("aliases", [])
                final_id = ent.id
            
            entity_dict = {
                "id": final_id,
                "canonical_name": ent.canonical_name,
                "type": ent.type,
                "aliases": current_aliases,
                "confidence": ent.confidence,
                "embedding": embedding_vec,
                "has_new_info": ent.has_new_info,
            }
            
            final_entities.append(entity_dict)
            
            if is_new or ent.has_new_info:
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
        
        self.publish_structure(final_entities, relationships, msg.id)

        if entities_needing_profile:
            asyncio.create_task(
                self._run_profile_updates(entities_needing_profile, msg.message.strip(), recent_context_list)
            )
        
    
    def publish_structure(self, normalized_entities: List[Dict], relationships: List[Dict], msg_id: int):
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
        
        self._send_batch_to_stream(msg_id, proto_ents, proto_rels)
        logger.info(f"Published structure for msg_{msg_id}: {len(proto_ents)} ents, {len(proto_rels)} rels")



