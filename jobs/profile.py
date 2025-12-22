import asyncio
from datetime import datetime
from functools import partial
import json
import re
import time
from typing import List, Optional, Set
from loguru import logger
from jobs.base import BaseJob, JobContext, JobNotifier, JobResult
from schema.common_pb2 import BatchMessage, Entity, MessageType
from redis import exceptions
from main.service import LLMService
from main.entity_resolve import EntityResolver
from main.prompts import get_profile_update_prompt

STREAM_KEY_PROFILE = "stream:profile" 

class ProfileRefinementJob(BaseJob):
    """
    Scans entities that have been 'touched' recently and updates their profiles
    using the sliding window of recent messages.
    
    Triggers:
    1. VOLUME: If >30 entities are dirty (ensures we catch them in the 75-msg window).
    2. TIME: If user is idle for >5 minutes and we have ANY dirty entities.
    """
    
    MSG_WINDOW = 75
    VOLUME_THRESHOLD = 30
    IDLE_THRESHOLD = 300
    
    def __init__(self, llm: LLMService, resolver: EntityResolver, executor):
        self.llm = llm
        self.resolver = resolver
        self.executor = executor

    @property
    def name(self) -> str:
        return "profile_refinement"

    async def should_run(self, ctx: JobContext) -> bool:
        dirty_key = f"dirty_entities:{ctx.user_name}"
        count = await ctx.redis.scard(dirty_key)
        
        if count >= self.VOLUME_THRESHOLD:
            return True
            
        if count > 0 and ctx.idle_seconds >= self.IDLE_THRESHOLD:
            return True
            
        return False

    async def execute(self, ctx: JobContext) -> JobResult:
        warning = "⚠️ **Deepening Profiles.** I am reading through recent conversations to update entity details. Please wait a moment for the best results."

        async with JobNotifier(ctx.redis, warning):
            dirty_key = f"dirty_entities:{ctx.user_name}"
            
            raw_ids = await ctx.redis.spop(dirty_key, 50)
            if not raw_ids:
                return JobResult(success=True, summary="No entities to update")
            
            entity_ids = [int(id_str) for id_str in raw_ids]
            

            sorted_set_key = f"recent_messages:{ctx.user_name}"
            recent_msg_ids = await ctx.redis.zrevrange(sorted_set_key, 0, self.MSG_WINDOW - 1)
            
            if not recent_msg_ids:
                await ctx.redis.sadd(dirty_key, *entity_ids)
                return JobResult(success=False, summary="No context messages found")

            msg_data_list = await ctx.redis.hmget(f"message_content:{ctx.user_name}", *recent_msg_ids)

            recent_context = []
            now = datetime.now()
            
            for msg_data in msg_data_list:
                if msg_data:
                    parsed = json.loads(msg_data)
                    raw = parsed['message']
                    ts = datetime.fromisoformat(parsed['timestamp'])
                    
                    delta = (now - ts).total_seconds()
                    if delta < 3600:
                        mins = int(delta // 60)
                        relative = f"{mins}m ago" if mins > 1 else "just now"
                    elif delta < 86400:
                        hours = int(delta // 3600)
                        relative = f"{hours}h ago"
                    else:
                        days = int(delta // 86400)
                        relative = f"{days}d ago"
                        
                    formatted = f"({relative}) {raw}"
                    recent_context.append((formatted, raw))
            
            updates = await self._run_updates(ctx, entity_ids, recent_context)
            
            if updates:
                await self._publish_updates(ctx, updates)

            return JobResult(
                success=True, 
                summary=f"Refined {len(updates)} profiles from {len(entity_ids)} candidates"
            )

    async def _run_updates(self, ctx: JobContext, entity_ids: List[int], recent_context: List[tuple]):
        current_msg_id = await ctx.redis.get("global:next_msg_id")
        current_msg_id = int(current_msg_id) if current_msg_id else 0
        
        semaphore = asyncio.Semaphore(5)

        async def update_single(ent_id: int) -> Optional[Entity]:
            async with semaphore:
                profile = self.resolver.entity_profiles.get(ent_id)
                if not profile:
                    return None
                
                canonical_name = profile.get("canonical_name", "Unknown")
                entity_type = profile.get("type", "unknown")
                existing_summary = profile.get("summary", "")

                mentions = self.resolver.get_mentions_for_id(ent_id)
                if not mentions:
                    return None

                pattern = re.compile(
                    r'\b(' + '|'.join(re.escape(m) for m in mentions) + r')\b', 
                    re.IGNORECASE
                )
                
                observations = [formatted for formatted, raw in recent_context if pattern.search(raw)]

                if not observations:
                    return None
                
                context_text = "\n".join(observations)
                system_prompt = get_profile_update_prompt(ctx.user_name)
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
                    self.executor,
                    partial(self.resolver.update_profile_summary, ent_id, new_summary)
                )
                
                logger.info(f"Refined profile for {canonical_name} (ID: {ent_id})")
                
                return Entity(
                    id=ent_id,
                    canonical_name=canonical_name,
                    type=entity_type,
                    summary=new_summary,
                    topic=profile.get("topic", "General"),
                    embedding=embedding,
                    last_profiled_msg_id=current_msg_id
                )
        
        results = await asyncio.gather(*[update_single(eid) for eid in entity_ids])
        return [r for r in results if r is not None]

    async def _publish_updates(self, ctx: JobContext, updates: List[Entity]):
        """
        Migrated from Context._send_batch_to_stream.
        Publishes the PROFILE_UPDATE batch to Redis Stream.
        """
        batch = BatchMessage(
            type=MessageType.PROFILE_UPDATE,
            list_ents=updates,
            list_relations=[]
        )
        
        serialized_data = batch.SerializeToString()
        
        try:
            batch_id = f"profile_job_{int(time.time())}"
            
            stream_payload = {
                'data': serialized_data,
                'batch_id': batch_id,
                'timestamp': str(time.time())
            }
            
            await ctx.redis.xadd(STREAM_KEY_PROFILE, stream_payload)
            logger.info(f"Published {len(updates)} profile updates to {STREAM_KEY_PROFILE}")

        except exceptions.RedisError as e:
            logger.error(f"Failed to publish profile updates: {e}")