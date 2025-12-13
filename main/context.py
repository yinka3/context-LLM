import asyncio
import time
from dotenv import load_dotenv
import os
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import json
import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI
from redis import exceptions
from redisclient import AsyncRedisClient
from typing import Dict, List, Set, Tuple, TypeVar, Type
from functools import partial
from schema.dtypes import *
from schema.common_pb2 import Entity, Relationship, BatchMessage, MessageType
from main.nlp_pipe import NLPPipeline
from main.entity_resolve import EntityResolver
from graph.memgraph import MemGraphStore
from main.prompts import *
from main.llm_trace import get_trace_logger
load_dotenv()

T = TypeVar('T', bound=BaseModel)
STREAM_KEY_AI_RESPONSE = "stream:ai_response"

LLM_CLIENT_INSTRUCT = lambda: instructor.from_openai(
    AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")),
                mode=instructor.Mode.JSON
)

LLM_CLIENT = lambda: AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
BATCH_SIZE = 5
PROFILE_INTERVAL = 15

STREAM_KEY_STRUCTURE = "stream:structure"
STREAM_KEY_PROFILE = "stream:profile"
BATCH_TIMEOUT_SECONDS = 60

class Context:

    def __init__(self, user_name: str, topics: List[str], redis_client, llm_client_instruct, llm_client):
        self.user_name: str = user_name
        self.active_topics: List[str] = topics
        self.session_emotions: List[str] = []
        
        self.redis_client: redis.Redis = redis_client 
        self.llm_instruct: AsyncOpenAI = llm_client_instruct
        self.llm_client: AsyncOpenAI = llm_client
        
        self.store: 'MemGraphStore' = None 
        self.cpu_executor: ThreadPoolExecutor = None
        self.nlp_pipe: 'NLPPipeline' = None
        self.ent_resolver: 'EntityResolver' = None

        self._background_tasks: Set[asyncio.Task] = set()
        self._batch_timer_task: asyncio.Task = None
        self._batch_processing_lock = asyncio.Lock()
        self._batch_queue_task: asyncio.Task = None

        self.trace_logger = get_trace_logger()

    @classmethod
    async def create(cls, user_name: str, topics: List[str] = ["General"]) -> "Context":
        redis_conn = AsyncRedisClient().get_client()
        llm = LLM_CLIENT()
        llm_instruct = LLM_CLIENT_INSTRUCT()
        
        instance = cls(user_name, topics, redis_conn, llm_instruct, llm)
        
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

    @staticmethod
    def _log_task_exception(task):
        if task.cancelled():
            return
        if exc := task.exception():
            logger.error(f"Background task failed: {exc}")
    
    def _fire_and_forget(self, coroutine):
        """Safely schedule background task with strong reference."""
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._log_task_exception)


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
            "canonical_name": user_name,
            "summary": f"The primary user named {user_name}",
            "type": "PERSON"
        }

        embedding_vector = await asyncio.get_running_loop().run_in_executor(
            self.cpu_executor, 
            partial(self.ent_resolver.add_entity, new_id, profile)
        )

        user_entity = Entity(
            id=new_id,
            canonical_name=user_name,
            type="PERSON",
            confidence=1.0,
            summary=profile["summary"],
            topic="Meta",
            embedding=embedding_vector
        )

        batch = BatchMessage(type=MessageType.SYSTEM_ENTITY)
        batch.list_ents.append(user_entity)

        try:
            await self.redis_client.xadd(STREAM_KEY_STRUCTURE, {'data': batch.SerializeToString()})
        except Exception as e:
            logger.error(f"Failed to push User Entity to stream: {e}")
        
    async def _flush_batch_timeout(self):
        try:
            await asyncio.sleep(BATCH_TIMEOUT_SECONDS)
            buffer_len = await self.redis_client.llen(f"buffer:{self.user_name}")
            if buffer_len > 0:
                logger.info(f"Batch timeout reached")
                await self._trigger_batch_process()
        except asyncio.CancelledError:
            pass
    
    async def _flush_batch_shutdown(self):

        logger.info("Initiating graceful shutdown of batch processor...")
        if self._batch_timer_task:
            self._batch_timer_task.cancel()
            self._batch_timer_task = None
        
        buffer_key = f"buffer:{self.user_name}"
        while await self.redis_client.llen(buffer_key) > 0:
            logger.info("Shutdown: Waiting for buffer to drain...")
            await asyncio.sleep(1)
            await self._trigger_batch_process() 

        async with self._batch_processing_lock:
            logger.info("Batch processing lock acquired - buffer is empty and active batches done.")

        if self._batch_queue_task and not self._batch_queue_task.done():
            logger.info("Stopping batch queue listener...")
            self._batch_queue_task.cancel()
            try:
                await self._batch_queue_task
            except asyncio.CancelledError:
                pass

        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background tasks...")
            await asyncio.wait(self._background_tasks, timeout=90)
        
        logger.info("All batches and background tasks completed")
    
    async def _trigger_batch_process(self):

        if self._batch_queue_task is None or self._batch_queue_task.done():
            self._batch_queue_task = asyncio.create_task(self._batch_a_queue())
    
    async def _batch_a_queue(self):

        while True:
            buffer_len = await self.redis_client.llen(f"buffer:{self.user_name}")
            
            if buffer_len > 0:
                await self.process_batch()
                if self._background_tasks:
                    await asyncio.wait(self._background_tasks, timeout=90)
            else:
                await asyncio.sleep(0.5)

    async def add(self, msg: MessageData):
        """Buffer message and trigger batch processing when ready."""
        msg.id = await self.get_next_msg_id()
        await self.add_to_redis(msg)

        buffer_key = f"buffer:{self.user_name}"
        await self.redis_client.rpush(buffer_key, json.dumps({
            "id": msg.id,
            "message": msg.message.strip(),
            "timestamp": msg.timestamp.isoformat()
        }))

        buffer_len = await self.redis_client.llen(buffer_key)
        
        if buffer_len >= BATCH_SIZE:
            if self._batch_timer_task:
                self._batch_timer_task.cancel()
                self._batch_timer_task = None
            
            await self._trigger_batch_process()
        elif buffer_len == 1:
            self._batch_timer_task = asyncio.create_task(self._flush_batch_timeout())

    
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

    
    def _build_profiling_prompt(
        self, 
        entity_name: str, 
        entity_type: str,
        existing_profile: Dict, 
        new_observation: str,
        conversation_context: List[str]
    ) -> Tuple[str, str]:
        
        system_prompt = get_profile_update_prompt()

        user_prompt_data = {
            "user_name": self.user_name,
            "entity_target": entity_name,
            "entity_type": entity_type,
            "existing_profile": existing_profile,
            "new_observation": new_observation,
            "recent_context": conversation_context,
            "valid_topics": self.active_topics
        }

        return system_prompt, json.dumps(user_prompt_data, indent=2)

    
    async def _call_slm(self, 
                        prompt: Tuple[str, str], 
                        response_model: Type[T]) -> T | None:
        
        system, user = prompt

        self.trace_logger.debug(
            f"MODEL: qwen/qwen-2.5-72b-instruct\n"
            f"SYSTEM PROMPT:\n{system}\n\n"
        )
        
        try:
            response = await self.llm_instruct.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_model=response_model,
                max_retries=2,
                temperature=0,
                extra_body={
                    "provider": {
                        "allow_fallbacks": True 
                    }
                }
            )
        
            self.trace_logger.debug(f"RESPONSE:\n{response.model_dump_json(indent=2)}")

            return response
        except Exception as e:
            self.trace_logger.error(f"GENERATION FAILED: {e}")
            logger.error(f"LLM Generation Failed: {e}")
            return None
    
    async def _call_reasoning(self, system: str, user: str) -> str | None:

        self.trace_logger.debug(
            f"MODEL: deepseek/deepseek-r1-distill-llama-70b\n"
            f"SYSTEM PROMPT:\n{system}\n\n"
        )

        try:
            response = await self.llm_client.chat.completions.create(
            model="deepseek/deepseek-v3.2",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.6,
            extra_body={
                    "provider": {
                        "allow_fallbacks": True 
                    }
                }
            )

            content = response.choices[0].message.content
            self.trace_logger.debug(f"RESPONSE:\n{content}")

            return content
        except Exception as e:
            self.trace_logger.error(f"GENERATION FAILED: {e}")
            logger.error(f"LLM Generation Failed: {e}")
            return None
    
    async def _call_formatter(self, reasoning_output: str, response_model: Type[T]) -> T | None:

        system = get_connection_formatter_prompt()

        try:
            response = await self.llm_instruct.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": reasoning_output}
                ],
                response_model=response_model,
                max_retries=2,
                temperature=0,
                extra_body={
                    "provider": {
                        "allow_fallbacks": True 
                    }
                }
            )

            return response
        except Exception as e:
            logger.error(f"Formatter failed: {e}")
            return None

    
    async def _send_batch_to_stream(self, entities: List[Entity], 
                                        relations: List[Relationship], type: MessageType, 
                                        stream_key: str = STREAM_KEY_STRUCTURE):
            batch = BatchMessage(
                type=type,
                list_ents=entities,
                list_relations=relations
            )
            try:
                await self.redis_client.xadd(stream_key, {'data': batch.SerializeToString()})
            except exceptions.RedisError as e:
                logger.error(f"Failed to push batch to {stream_key}: {e}")
    

    async def _disambiguate(
        self,
        resolution_result: Dict,
        messages: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Disambiguate ambiguous mentions and deduplicate new entities.
        """
        resolved = resolution_result["resolved"]
        ambiguous = resolution_result["ambiguous"]
        new = resolution_result["new"]
        
        entity_registry: Dict[str, Dict] = {}
        
        for mention, data in resolved.items():
            canonical = data["canonical_name"]
            if canonical not in entity_registry:
                entity_registry[canonical] = {
                    "id": data["id"],
                    "type": data["type"],
                    "mentions": [mention]
                }
            else:
                if mention not in entity_registry[canonical]["mentions"]:
                    entity_registry[canonical]["mentions"].append(mention)
        
        needs_disambiguation = bool(ambiguous) or len(new) > 1
        
        if not needs_disambiguation:
            if new:
                mention, data = next(iter(new.items()))
                new_id = await self.get_next_ent_id()
                entity_registry[mention] = {
                    "id": new_id,
                    "type": data["type"],
                    "mentions": [mention],
                    "is_new": True
                }
            return entity_registry
        
        messages_text = "\n".join([f"{m['id']}: \"{m['message']}\"" for m in messages])
        user_profile = self.ent_resolver.entity_profiles.get(1)
        system_user_info = None
        if user_profile:
             system_user_info = {
                 "id": 1,
                 "canonical_name": user_profile.get("canonical_name"),
                 "aliases": user_profile.get("aliases", [])
             }

        system_prompt = get_disambiguation_prompt(messages_text, self.user_name)
        
        user_prompt_data = {
            "system_user_context": system_user_info, 
            "ambiguous_mentions": [],
            "new_mentions": []
        }
        
        for mention, data in ambiguous.items():
            candidates_info = [
                {
                    "id": c["id"],
                    "canonical_name": c["profile"].get("canonical_name", "Unknown"),
                    "type": c["profile"].get("type", "UNKNOWN"),
                    "summary": c["profile"].get("summary", "No summary available.")
                }
                for c in data["candidates"]
            ]
            user_prompt_data["ambiguous_mentions"].append({
                "mention": mention,
                "type": data["type"],
                "candidates": candidates_info
            })
        
        for mention, data in new.items():
            user_prompt_data["new_mentions"].append({
                "mention": mention,
                "type": data["type"]
            })
        
        disambiguation_response = await self._call_slm(
            prompt=(system_prompt, json.dumps(user_prompt_data, indent=2)),
            response_model=DisambiguationResponse
        )
        
        if not disambiguation_response:
            logger.error("Disambiguation LLM call failed, falling back to treating all as new")

            for mention, data in ambiguous.items():
                new_id = await self.get_next_ent_id()
                entity_registry[mention] = {
                    "id": new_id,
                    "type": data["type"],
                    "mentions": [mention],
                    "is_new": True
                }
            for mention, data in new.items():
                new_id = await self.get_next_ent_id()
                entity_registry[mention] = {
                    "id": new_id,
                    "type": data["type"],
                    "mentions": [mention],
                    "is_new": True
                }
            return entity_registry
        
        for resolution in disambiguation_response.ambiguous_resolutions:
            mention = resolution.mention
            if mention not in ambiguous:
                logger.warning(f"LLM returned mention '{mention}' not found in ambiguous list. Skipping.")
                continue

            if resolution.is_new:
                if mention not in new:
                    new[mention] = {"type": ambiguous[mention]["type"]}
            else:
                canonical = resolution.canonical_name
                final_id = resolution.resolved_id
                if final_id is None and canonical:

                    for cand in ambiguous[mention]["candidates"]:
                        if cand["profile"].get("canonical_name") == canonical:
                            final_id = cand["id"]
                            break

                    if canonical == self.user_name:
                        final_id = 1
                
                if final_id is None:
                    logger.warning(f"LLM resolved '{mention}' to '{canonical}' but ID is Null. Treating as New.")
                    if mention not in new:
                        new[mention] = {"type": ambiguous[mention]["type"]}
                    continue

                if canonical not in entity_registry:
                    entity_registry[canonical] = {
                        "id": resolution.resolved_id,
                        "type": ambiguous[mention]["type"],
                        "mentions": [mention]
                    }
                else:
                    if mention not in entity_registry[canonical]["mentions"]:
                        entity_registry[canonical]["mentions"].append(mention)
        
        for group in disambiguation_response.new_entity_groups:
            new_id = await self.get_next_ent_id()
            
            stub_profile = {
                "canonical_name": group.canonical_name,
                "aliases": [m for m in group.mentions if m != group.canonical_name],
                "summary": "",
                "type": group.type
            }
            
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                self.cpu_executor,
                partial(self.ent_resolver.add_entity, new_id, stub_profile)
            )
            
            entity_registry[group.canonical_name] = {
                "id": new_id,
                "type": group.type,
                "mentions": group.mentions,
                "is_new": True,
                "embedding": embedding
            }
        
        return entity_registry
    
    async def _extract_connections(
        self,
        entity_registry: Dict[str, Dict],
        messages: List[Dict]
    ) -> ConnectionExtractionResponse | None:
        """
        Extract meaningful connections between entities in messages.
        """
    
        candidate_entities = [
            {"name": name, "type": data["type"], "mentions": data["mentions"]}
            for name, data in entity_registry.items()
        ]
        
        # Pass 1: Reasoning
        reasoning_prompt = get_connection_reasoning_prompt(self.user_name)
        user_content = json.dumps({"candidate_entities": candidate_entities, "messages": messages})
        reasoning_output = await self._call_reasoning(reasoning_prompt, user_content)
        
        if not reasoning_output:
            return None
        
        result = await self._call_formatter(reasoning_output, ConnectionExtractionResponse)
        return result
    

    async def _get_buffered_messages(self) -> List[Dict]:
        """Atomically grab batch from buffer."""
        buffer_key = f"buffer:{self.user_name}"
        pipe = self.redis_client.pipeline()
        pipe.lrange(buffer_key, 0, BATCH_SIZE - 1)
        pipe.ltrim(buffer_key, BATCH_SIZE, -1)
        results = await pipe.execute()
        
        raw_messages = results[0]
        return [json.loads(m) for m in raw_messages] if raw_messages else []


    async def _extract_mentions_batch(self, messages: List[Dict]) -> Dict[str, str]:
        """Phase 1: Parallel NLP extraction across all messages."""
        loop = asyncio.get_running_loop()
        
        mention_tasks = [
            loop.run_in_executor(self.cpu_executor, self.nlp_pipe.extract_mentions, m["message"], 0.80)
            for m in messages
        ]
        emotion_tasks = [
            loop.run_in_executor(self.cpu_executor, self.nlp_pipe.analyze_emotion, m["message"])
            for m in messages
        ]

        all_mentions_results = await asyncio.gather(*mention_tasks)
        all_emotions = await asyncio.gather(*emotion_tasks)

        all_unique_mentions: Dict[str, str] = {}
        for mentions in all_mentions_results:
            for mention_text, mention_type in mentions:
                if mention_text not in all_unique_mentions:
                    all_unique_mentions[mention_text] = mention_type

        for emotions in all_emotions:
            if emotions:
                dominant = max(emotions, key=lambda x: x["score"])
                self.session_emotions.append(dominant["label"])

        return all_unique_mentions


    async def _resolve_mentions(self, mentions: Dict[str, str], messages: List[Dict]) -> Dict[str, Dict]:
        """Phase 2: Resolve all unique mentions against existing entities."""
        if not mentions:
            return {"resolved": {}, "ambiguous": {}, "new": {}}

        loop = asyncio.get_running_loop()
        context = " ".join(m["message"] for m in messages)
        
        tasks = [
            loop.run_in_executor(
                self.cpu_executor,
                partial(self.ent_resolver.resolve, text=mention, context=context)
            )
            for mention in mentions
        ]
        results = await asyncio.gather(*tasks)
        
        resolved = {}
        ambiguous = {}
        new = {}
        
        for mention_text, result in zip(mentions.keys(), results):
            mention_type = mentions[mention_text]
            
            if result["resolved"]:
                resolved[mention_text] = {
                    "id": result["resolved"]["id"],
                    "type": mention_type,
                    "canonical_name": result["resolved"]["profile"]["canonical_name"]
                }
            elif result["ambiguous"]:
                ambiguous[mention_text] = {
                    "type": mention_type,
                    "candidates": result["ambiguous"]
                }
            elif result["new"]:
                new[mention_text] = {
                    "type": mention_type
                }
        
        return {"resolved": resolved, "ambiguous": ambiguous, "new": new}


    async def _run_profile_updates(self, 
        entities_list: List[Dict], 
        context_text: str,
        recent_context_list: List[str],
        checkpoint_msg_id: int):
        
        """Background task to run Prompt 2 for specific entities"""

        async def update_single_entity(ent):
            ent_id = ent["id"]
            existing_profile = self.ent_resolver.entity_profiles.get(ent_id, {})
            old_summary = existing_profile.get("summary", "No information provided")

            logger.info(f"Profile update for {ent_id}: old_summary='{old_summary[:50] if old_summary else 'EMPTY'}'")
            
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
            

            new_summary = updated_profile.summary
        
            if new_summary == old_summary:
                return None

            loop = asyncio.get_running_loop()
            embedding_array = await loop.run_in_executor(
                self.cpu_executor,
                partial(
                    self.ent_resolver.embedding_model.encode,
                    [new_summary]
                )
            )
            new_embedding = embedding_array[0].tolist() if embedding_array is not None else []
                    
            return Entity(
                id=ent_id,
                canonical_name=updated_profile.canonical_name,
                type=ent["type"],
                summary=new_summary,
                topic=updated_profile.topic,
                embedding=new_embedding,
                last_profiled_msg_id=checkpoint_msg_id
            )
        
        tasks = [update_single_entity(ent) for ent in entities_list]
        results = await asyncio.gather(*tasks)
        updates_for_graph = [r for r in results if r is not None]
        
        if updates_for_graph:
            await self._send_batch_to_stream(
                entities=updates_for_graph,
                relations=[],
                type=MessageType.PROFILE_UPDATE,
                stream_key=STREAM_KEY_PROFILE
            )
            logger.info(f"Pushed Profile Batch {checkpoint_msg_id} to {STREAM_KEY_PROFILE}")


    async def process_batch(self):
        async with self._batch_processing_lock:
            logger.info("Starting batch processing...")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.cpu_executor, self.ent_resolver._init_from_db)
            logger.info("Refreshed Entity resolver from DB")

            messages = await self._get_buffered_messages()
            if not messages:
                return

            logger.info(f"Processing batch of {len(messages)} messages: {[m['id'] for m in messages]}")
            current_batch_max_id = max([m['id'] for m in messages])

            mentions = await self._extract_mentions_batch(messages)
            if not mentions:
                logger.info("No mentions found in batch, skipping LLM calls")
                return
            
            resolution_result = await self._resolve_mentions(mentions, messages)
            entity_registry = await self._disambiguate(resolution_result, messages)

            if not entity_registry:
                logger.info("No entities after disambiguation, skipping connection extraction")
                return
        
            connection_result = await self._extract_connections(entity_registry, messages)

            if not connection_result:
                logger.error("Connection extraction failed")
                return

            await self._publish_batch(entity_registry, connection_result)

            entities_needing_profile = []

            for name, data in entity_registry.items():
                ent_id = data["id"]
                curr_profile = self.ent_resolver.entity_profiles.get(ent_id, {})
                raw_last_profiled = curr_profile.get("last_profiled_msg_id")
                last_profiled_id = raw_last_profiled if raw_last_profiled is not None else 0

                is_new = data.get("is_new", False)
                update_time = (current_batch_max_id - last_profiled_id) >= PROFILE_INTERVAL

                if is_new or update_time:
                    profile_data = data.copy()
                    profile_data["canonical_name"] = name 
                    entities_needing_profile.append(profile_data)
                    logger.info(f"Triggering Profile for {name} (ID: {ent_id}). New={is_new}, Gap={current_batch_max_id - last_profiled_id}")
                else:
                    logger.debug(f"Skipping Profile for {name}: Only {current_batch_max_id - last_profiled_id} msgs since last update.")
            
            if entities_needing_profile:
                recent_context = await self.get_recent_context(num_messages=PROFILE_INTERVAL)
                batch_context = " ".join(m["message"] for m in messages)
                
                self._fire_and_forget(
                    self._run_profile_updates(
                        entities_needing_profile, 
                        batch_context, 
                        recent_context, 
                        current_batch_max_id
                    )
                )

            logger.info(f"Batch complete: Max ID: {current_batch_max_id}, {len(entity_registry)} entities, {len(messages)} messages processed")
        
    
    async def _publish_batch(
        self,
        entity_registry: Dict[str, Dict],
        extraction_result: ConnectionExtractionResponse
        ):
        """
        Publish entities and relationships to graph via Redis stream.
        """
    
        entity_lookup = {
            name.lower(): {
                "id": data["id"],
                "canonical_name": name,
                "type": data["type"],
                "summary": "",
                "topic": "General",
                "embedding": data.get("embedding", [])
            }
            for name, data in entity_registry.items()
        }
        
        for msg_result in extraction_result.message_results:
            msg_id = msg_result.message_id
            
            mentioned_entities = set()
            for pair in msg_result.entity_pairs:
                mentioned_entities.add(pair.entity_a.lower())
                mentioned_entities.add(pair.entity_b.lower())
            

            proto_ents = []
            seen_ids = set()
            for entity_key in mentioned_entities:
                ent = entity_lookup.get(entity_key)
      
                if ent and ent["id"] not in seen_ids and ent["type"]:
                    proto_ents.append(Entity(
                        id=ent["id"],
                        canonical_name=ent["canonical_name"],
                        type=ent["type"],
                        confidence=1.0,
                        summary="",
                        topic=ent["topic"],
                        embedding=ent["embedding"]
                    ))
                    seen_ids.add(ent["id"])
            

            proto_rels = []
            for pair in msg_result.entity_pairs:
                ent_a = entity_lookup.get(pair.entity_a.lower())
                ent_b = entity_lookup.get(pair.entity_b.lower())
                
                if ent_a and ent_b:
                    proto_rels.append(Relationship(
                        entity_a=ent_a["canonical_name"],
                        entity_b=ent_b["canonical_name"],
                        confidence=pair.confidence,
                        message_id=msg_id
                    ))
                else:
                    logger.warning(f"Skipping pair: {pair.entity_a} - {pair.entity_b} (entity not found)")
        
            await self._send_batch_to_stream(
                entities=proto_ents,
                relations=proto_rels,
                type=MessageType.USER_MESSAGE,
                stream_key=STREAM_KEY_STRUCTURE
            )
    
    async def shutdown(self):

        await self._flush_batch_shutdown()

        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background tasks...")
            await asyncio.wait(self._background_tasks, timeout=60)
        
        loop = asyncio.get_running_loop()
        candidates = await loop.run_in_executor(
            self.cpu_executor,
            self.ent_resolver.detect_merge_candidates
        )

        if candidates:
            logger.info(f"Detected {len(candidates)} merge candidates at shutdown")

        if self.session_emotions:
            from collections import Counter
            counts = Counter(self.session_emotions)
            top_two = counts.most_common(2)
            
            primary, primary_count = top_two[0]
            secondary, secondary_count = top_two[1] if len(top_two) > 1 else ("neutral", 0)
            
            self.store.log_daily_mood(
                primary=primary,
                primary_count=primary_count,
                secondary=secondary,
                secondary_count=secondary_count,
                total=len(self.session_emotions)
            )
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.redis_client:
            await self.redis_client.aclose()
    



