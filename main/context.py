import asyncio
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
        instance.cpu_executor = cpu_executor
        
        loop = asyncio.get_running_loop()
        
        instance.nlp_pipe = await loop.run_in_executor(
            instance.cpu_executor, 
            partial(NLPPipeline, llm=instance.llm)
        )
        
        instance.ent_resolver = ent_resolver

        await instance._get_or_create_user_entity(user_name)

        instance.batch_processor = BatchProcessor(
            redis_client=redis_conn,
            llm=instance.llm,
            ent_resolver=instance.ent_resolver,
            nlp_pipe=instance.nlp_pipe,
            cpu_executor=instance.cpu_executor,
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
                    self.cpu_executor,
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
            self.cpu_executor,
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


