import asyncio
import re
import time
from dotenv import load_dotenv
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import json
from main.processor import BatchProcessor
from main.service import LLMService
from redis import exceptions
from redisclient import AsyncRedisClient
from typing import List, Set, Tuple
from functools import partial
from schema.dtypes import *
from schema.common_pb2 import Entity, Relationship, BatchMessage, MessageType
from main.nlp_pipe import NLPPipeline
from main.entity_resolve import EntityResolver
from graph.memgraph import MemGraphStore
from main.prompts import *
from main.llm_trace import get_trace_logger
from schedule.scheduler import Scheduler
from schedule.merger import MergeDetectionJob
load_dotenv()

STREAM_KEY_AI_RESPONSE = "stream:ai_response"

BATCH_SIZE = 5
PROFILE_INTERVAL = 15

STREAM_KEY_STRUCTURE = "stream:structure"
STREAM_KEY_PROFILE = "stream:profile"
BATCH_TIMEOUT_SECONDS = 180.0

class Context:

    def __init__(self, user_name: str, topics: List[str], redis_client):
        self.user_name: str = user_name
        self.active_topics: List[str] = topics
        self.session_emotions: List[str] = []
        self._session_entity_ids: Set[int] = set()
        self.scheduler: Scheduler = None
        self.redis_client: redis.Redis = redis_client
        self.llm: LLMService = None
        
        self.store: 'MemGraphStore' = None
        self.nlp_pipe: 'NLPPipeline' = None
        self.ent_resolver: 'EntityResolver' = None

        self.executor: ThreadPoolExecutor = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._batch_timer_task: asyncio.Task = None
        self._batch_processing_lock = asyncio.Lock()
        self._batch_queue_task: asyncio.Task = None
        self.batch_processor: BatchProcessor = None

        self.trace_logger = get_trace_logger()

    @classmethod
    async def create(
        cls,
        user_name: str,
        ent_resolver: EntityResolver,
        store: MemGraphStore,
        cpu_executor: ThreadPoolExecutor,
        topics: List[str] = ["General"]
    ) -> "Context":
        redis_conn = AsyncRedisClient().get_client()
        
        instance = cls(user_name, topics, redis_conn)
        instance.llm = LLMService(trace_logger=get_trace_logger())
        
        instance.store = store
        instance.executor = cpu_executor
        
        loop = asyncio.get_running_loop()
        
        instance.nlp_pipe = await loop.run_in_executor(
            instance.executor, 
            partial(NLPPipeline, llm=instance.llm)
        )
        
        instance.ent_resolver = ent_resolver

        await instance._get_or_create_user_entity(user_name)

        instance.batch_processor = BatchProcessor(
            redis_client=redis_conn,
            llm=instance.llm,
            ent_resolver=instance.ent_resolver,
            nlp_pipe=instance.nlp_pipe,
            cpu_executor=instance.executor,
            user_name=user_name,
            active_topics=topics,
            get_next_ent_id=instance.get_next_ent_id,
        )

        instance.scheduler = Scheduler(user_name)
        instance.scheduler.register(
            MergeDetectionJob(instance.ent_resolver, instance.store)
        )
        await instance.scheduler.start()
        
        return instance

    @staticmethod
    def _log_task_exception(task):
        if task.cancelled():
            return
        if exc := task.exception():
            logger.error(f"Background task failed: {exc}")
    
    def _fire_and_forget(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._log_task_exception)


    async def get_next_msg_id(self) -> int:
        return await self.redis_client.incr("global:next_msg_id")

    async def get_next_ent_id(self) -> int:
        return await self.redis_client.incr("global:next_ent_id")
    
    def _format_relative_time(self, now: datetime, ts: datetime) -> str:
        delta = now - ts
        seconds = delta.total_seconds()
        
        if seconds < 3600:
            mins = int(seconds // 60)
            return f"{mins}m ago" if mins > 1 else "just now"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours}h ago"
        elif seconds < 604800:
            days = int(seconds // 86400)
            return f"{days}d ago"
        else:
            weeks = int(seconds // 604800)
            return f"{weeks}w ago"


    async def _get_or_create_user_entity(self, user_name: str):
        loop = asyncio.get_running_loop()

        entity_id = await loop.run_in_executor(
                    self.executor,
                    self.ent_resolver.get_id,
                    user_name
                )

        if entity_id:
            logger.info(f"User {user_name} recognized.")
            return entity_id
        
        logger.info(f"Creating new USER entity for {user_name}")
        new_id = await self.get_next_ent_id()
        
        profile = {
            "canonical_name": user_name,
            "summary": f"The primary user named {user_name}",
            "type": "person",
            "topic": "Personal"
        }

        embedding = await loop.run_in_executor(
            self.executor,
            partial(self.ent_resolver.register_entity, new_id, user_name, [user_name], "person", "Personal")
        )
        
        self.ent_resolver.entity_profiles[new_id]["summary"] = profile["summary"]
        self._session_entity_ids.add(new_id)
        user_entity = Entity(
            id=new_id,
            canonical_name=user_name,
            type="person",
            confidence=1.0,
            summary=profile["summary"],
            topic="Personal",
            embedding=embedding,
            aliases=[user_name]
        )

        batch = BatchMessage(type=MessageType.SYSTEM_ENTITY)
        batch.list_ents.append(user_entity)

        try:
            await self.redis_client.xadd(STREAM_KEY_STRUCTURE, {'data': batch.SerializeToString()})
        except Exception as e:
            logger.error(f"Failed to push User Entity to stream: {e}")
        
        return new_id
        
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
        wait_count = 0
        while await self.redis_client.llen(buffer_key) > 0:
            if wait_count % 20 == 0:
                logger.info(f"Shutdown: Waiting for buffer to drain... ({wait_count}s)")
            wait_count += 1
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
            
            if buffer_len >= BATCH_SIZE:
                await self.process_batch()
                if self._background_tasks:
                    await asyncio.wait(self._background_tasks, timeout=180)
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
        await self.scheduler.record_activity()
        
        if buffer_len >= BATCH_SIZE:
            if self._batch_timer_task:
                self._batch_timer_task.cancel()
                self._batch_timer_task = None
            
            await self._trigger_batch_process()
        elif buffer_len == 1:
            self._batch_timer_task = asyncio.create_task(self._flush_batch_timeout())

    
    async def get_recent_context(self, num_messages: int = 10) -> List[Tuple[str, str]]:
        """Returns list of (formatted_message, raw_message) tuples."""
        sorted_set_key = f"recent_messages:{self.user_name}"
        recent_msg_ids = await self.redis_client.zrevrange(sorted_set_key, 0, num_messages-1)
        
        if not recent_msg_ids:
            return []
        
        recent_msg_ids.reverse()
        
        msg_data_list = await self.redis_client.hmget(
            f"message_content:{self.user_name}", 
            *recent_msg_ids
        )
        
        results = []
        now = datetime.now()
        
        for msg_data in msg_data_list:
            if msg_data:
                parsed = json.loads(msg_data)
                raw = parsed['message']
                ts = datetime.fromisoformat(parsed['timestamp'])
                relative = self._format_relative_time(now, ts)
                results.append((f"({relative}) {raw}", raw))

        return results


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

    
    async def _send_batch_to_stream(self, entities: List[Entity], 
                                        relations: List[Relationship], type: MessageType, 
                                        stream_key: str = STREAM_KEY_STRUCTURE):
        batch = BatchMessage(
            type=type,
            list_ents=entities,
            list_relations=relations
        )
        serialized_data = batch.SerializeToString()
    
        try:
            batch_id = f"batch_{int(time.time())}_{os.urandom(4).hex()}"
            snapshot_key = f"snapshot:{batch_id}"

            await self.redis_client.setex(snapshot_key, 3600, serialized_data)

            stream_payload = {
                'data': serialized_data,
                'batch_id': batch_id,
                'timestamp': str(time.time())
            }
            
            await self.redis_client.xadd(stream_key, stream_payload)
            logger.info(f"Published batch {batch_id} to stream (Snapshot secured)")

        except exceptions.RedisError as e:
            logger.critical(f"Redis WRITE failure in _publish_batch. Data potentially lost: {e}")


    async def _run_session_profile_updates(self):

        logger.info(f"Profiling {len(self._session_entity_ids)} session entities...")
        
        recent_context = await self.get_recent_context(num_messages=75)
        
        current_msg_id = await self.redis_client.get("global:next_msg_id")
        current_msg_id = int(current_msg_id) if current_msg_id else 0
        
        semaphore = asyncio.Semaphore(5)

        async def update_single(ent_id: int) -> Optional[Entity]:
            async with semaphore:
                profile = self.ent_resolver.entity_profiles.get(ent_id)
                if not profile:
                    return None
                
                canonical_name = profile.get("canonical_name", "Unknown")
                entity_type = profile.get("type", "unknown")
                existing_summary = profile.get("summary", "")


                mentions = self.ent_resolver.get_mentions_for_id(ent_id)
                if not mentions:
                    logger.debug(f"No mentions for {canonical_name}, skipping profile")
                    return None

                pattern = re.compile(
                    r'\b(' + '|'.join(re.escape(m) for m in mentions) + r')\b', 
                    re.IGNORECASE
                )
                
                observations = [formatted for formatted, raw in recent_context if pattern.search(raw)]

                if not observations:
                    logger.debug(f"No observations for {canonical_name}, skipping profile")
                    return None
                
                context_text = "\n".join(observations)
                system_prompt = get_profile_update_prompt(self.user_name)
                user_content = json.dumps({
                    "entity_name": canonical_name,
                    "entity_type": entity_type,
                    "existing_summary": existing_summary,
                    "new_observations": context_text,
                    "known_aliases": mentions
                }, indent=2)
                
                new_summary = await self.llm.call_reasoning(system_prompt, user_content)
                
                if not new_summary or new_summary == existing_summary:
                    return None
                
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    self.cpu_executor,
                    partial(self.ent_resolver.update_profile_summary, ent_id, new_summary)
                )
                
                logger.info(f"Profiled entity {ent_id}: {canonical_name}")
                
                return Entity(
                    id=ent_id,
                    canonical_name=canonical_name,
                    type=entity_type,
                    summary=new_summary,
                    topic=profile.get("topic", "General"),
                    embedding=embedding,
                    last_profiled_msg_id=current_msg_id
                )
        
        results = await asyncio.gather(*[update_single(ent_id) for ent_id in self._session_entity_ids])
        updates_for_graph = [r for r in results if r is not None]
        
        if updates_for_graph:
            await self._send_batch_to_stream(
                entities=updates_for_graph,
                relations=[],
                type=MessageType.PROFILE_UPDATE,
                stream_key=STREAM_KEY_PROFILE
            )
            logger.info(f"Sent {len(updates_for_graph)} profile updates to stream")


    async def process_batch(self):
        async with self._batch_processing_lock:
            logger.info("Starting batch processing...")
            
            buffer_key = f"buffer:{self.user_name}"
            messages = await self.batch_processor.get_buffered_messages(buffer_key, BATCH_SIZE)
            
            if not messages:
                return
            
            result = await self.batch_processor.run(messages)
            
            if not result.success:
                await self.batch_processor.move_to_dead_letter(messages, result.error)
            else:
                self._session_entity_ids.update(result.entity_ids)
                self.session_emotions.extend(result.emotions)
                
                if result.extraction_result:
                    await self._publish_batch(
                        result.entity_ids,
                        result.new_entity_ids,
                        result.alias_updated_ids,
                        result.extraction_result
                    )
            
            await self.redis_client.ltrim(buffer_key, len(messages), -1)
            logger.debug(f"Trimmed {len(messages)} from {buffer_key}")

    
    async def _publish_batch(
        self,
        entity_ids: List[int],
        new_entity_ids: Set[int],
        alias_updated_ids: Set[int],
        extraction_result: ConnectionExtractionResponse
        ):
        """
        Publish entities and relationships to graph via Redis stream.
        """
    
        entity_lookup = {}
        for ent_id in entity_ids:
            profile = self.ent_resolver.entity_profiles.get(ent_id)
            if profile:
                canonical = profile["canonical_name"]
                entity_lookup[canonical.lower()] = {
                    "id": ent_id,
                    "canonical_name": canonical,
                    "type": profile.get("type"),
                    "topic": profile.get("topic", "General")
                }
                for mention in self.ent_resolver.get_mentions_for_id(ent_id):
                    entity_lookup[mention.lower()] = entity_lookup[canonical.lower()]
        

        proto_ents = []
        for ent_id in new_entity_ids:
            profile = self.ent_resolver.entity_profiles.get(ent_id)
            if profile:
                embedding = self.ent_resolver.get_embedding_for_id(ent_id)
                aliases = self.ent_resolver.get_mentions_for_id(ent_id)
                proto_ents.append(Entity(
                    id=ent_id,
                    canonical_name=profile["canonical_name"],
                    type=profile.get("type", ""),
                    confidence=1.0,
                    summary="",
                    topic=profile.get("topic", "General"),
                    embedding=embedding,
                    aliases=aliases
                ))
        
        if alias_updated_ids:
            for ent_id in alias_updated_ids:
                if ent_id in new_entity_ids:
                    continue
                profile = self.ent_resolver.entity_profiles.get(ent_id)
                if profile:
                    aliases = self.ent_resolver.get_mentions_for_id(ent_id)
                    proto_ents.append(Entity(
                        id=ent_id,
                        canonical_name=profile["canonical_name"],
                        confidence=1.0,
                        aliases=aliases
                    ))

        proto_rels = []
        for msg_result in extraction_result.message_results:
            msg_id = msg_result.message_id
            
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
        
        if self._session_entity_ids:
            await self._run_session_profile_updates()
        
        logger.info("Waiting for graph consumer to sync...")
        await asyncio.sleep(20)
        
        await self.scheduler.stop()
        
        mentions = self.ent_resolver.get_mentions()
        if mentions:
            await self.redis_client.hset("entity_mentions", mapping=mentions)
            logger.info(f"Persisted {len(mentions)} mention mappings to Redis")

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
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.redis_client:
            await self.redis_client.aclose()