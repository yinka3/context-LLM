import asyncio
from datetime import datetime, timezone
from functools import partial
import json
import re
from typing import Dict, List, Optional
from loguru import logger
from db.memgraph import MemGraphStore
from jobs.base import BaseJob, JobContext, JobNotifier, JobResult
from main.service import LLMService
from main.entity_resolve import EntityResolver
from main.prompts import get_profile_update_prompt


class ProfileRefinementJob(BaseJob):
    """
    Scans entities that have been 'touched' recently and updates their profiles
    using the sliding window of recent messages.
    
    Triggers:
    1. VOLUME: If >=5 entities are dirty (ensures we catch them in the 75-msg window).
    2. TIME: If user is idle for >5 minutes and we have ANY dirty entities.
    """
    
    MSG_WINDOW = 75
    VOLUME_THRESHOLD = 5
    IDLE_THRESHOLD = 300
    USER_IDLE_THRESHOLD = 600
    USER_MSG_COUNT = 45
    
    def __init__(self, llm: LLMService, resolver: EntityResolver, store: MemGraphStore, executor):
        self.llm = llm
        self.resolver = resolver
        self.store = store
        self.executor = executor

    @property
    def name(self) -> str:
        return "profile_refinement"

    async def should_run(self, ctx: JobContext) -> bool:
        dirty_key = f"dirty_entities:{ctx.user_name}"
        return await ctx.redis.scard(dirty_key) > 0
    
    async def _maybe_refine_user(self, ctx: JobContext, dirty_count: int) -> bool:
        """
        Check conditions and trigger user profile refinement if needed.
        Returns True if refinement ran.
        """
        ran_key = f"user_profile_ran:{ctx.user_name}"
        if await ctx.redis.get(ran_key):
            return False
        
        if dirty_count < self.VOLUME_THRESHOLD and ctx.idle_seconds < self.USER_IDLE_THRESHOLD:
            return False
        
        user_id = self.resolver.get_id(ctx.user_name)
        if not user_id:
            logger.warning(f"User entity {ctx.user_name} not found in resolver")
            return False
        
        profile = self.resolver.entity_profiles.get(user_id)
        if not profile:
            logger.warning(f"User profile {user_id} not found")
            return False
        
        success = await self._refine_user_profile(ctx, user_id, profile)

        await ctx.redis.setex(ran_key, 300, "true")
        
        return success
    
    async def _get_conversation_context(self, ctx: JobContext, num_turns: int) -> List[Dict]:
        """Fetch recent conversation with both user and STELLA turns."""
        sorted_key = f"recent_conversation:{ctx.user_name}"
        conv_key = f"conversation:{ctx.user_name}"
        
        turn_ids = await ctx.redis.zrevrange(sorted_key, 0, num_turns - 1)
        if not turn_ids:
            return []
        
        turn_ids.reverse()
        turn_data = await ctx.redis.hmget(conv_key, *turn_ids)
        
        results = []
        now = datetime.now(timezone.utc)
        
        for data in turn_data:
            if data:
                parsed = json.loads(data)
                ts = datetime.fromisoformat(parsed['timestamp'])
                delta = (now - ts).total_seconds()
                
                if delta < 3600:
                    relative = f"{int(delta // 60)}m ago" if delta >= 60 else "just now"
                elif delta < 86400:
                    relative = f"{int(delta // 3600)}h ago"
                else:
                    relative = f"{int(delta // 86400)}d ago"
                
                role_label = "User" if parsed["role"] == "user" else "STELLA"
                results.append({
                    "role": parsed["role"],
                    "role_label": role_label,
                    "content": parsed["content"],
                    "formatted": f"({relative}) [{role_label}]: {parsed['content']}",
                    "raw": parsed["content"]
                })
        
        return results

    async def execute(self, ctx: JobContext) -> JobResult:
        warning = "⚠️ **Deepening Profiles.** I am reading through recent conversations to update entity details. Please wait a moment for the best results."

        async with JobNotifier(ctx.redis, warning):
            dirty_key = f"dirty_entities:{ctx.user_name}"
            dirty_count = await ctx.redis.scard(dirty_key)
            raw_ids = await ctx.redis.spop(dirty_key, dirty_count)
            await ctx.redis.delete(dirty_key)
            
            user_id = self.resolver.get_id(ctx.user_name)
            entity_ids = [int(id_str) for id_str in raw_ids if int(id_str) != user_id] if raw_ids else []
            
            updates = []
            
            if entity_ids:
                conversation = await self._get_conversation_context(ctx, self.MSG_WINDOW * 2)
        
                if not conversation:
                    await ctx.redis.sadd(dirty_key, *[str(eid) for eid in entity_ids])
                    return JobResult(success=False, summary="No context found")
                
                updates = await self._run_updates(ctx, entity_ids, conversation)
                
                if updates:
                    await self._write_updates(updates)
            
            user_refined = await self._maybe_refine_user(ctx, dirty_count)
            
            parts = []
            if updates:
                parts.append(f"Refined {len(updates)} profiles")
            if user_refined:
                parts.append(f"refined {ctx.user_name}")
            
            summary = ", ".join(parts) if parts else "No profiles to update"

            await ctx.redis.setex(
                f"profile_complete:{ctx.user_name}",
                300,
                str(datetime.now(timezone.utc).timestamp())
            )
            
            return JobResult(success=True, summary=summary)
    
    def _extract_summary(self, response: str) -> str:
        """Extract summary from reasoning+summary response."""
        if "<summary>" in response and "</summary>" in response:
            start = response.index("<summary>") + len("<summary>")
            end = response.index("</summary>")
            return response[start:end].strip()
        return response.strip()
    
    async def _refine_user_profile(self, ctx: JobContext, user_id: int, profile: dict) -> bool:
        """Execute user profile refinement."""
        conversation = await self._get_conversation_context(ctx, self.USER_MSG_COUNT * 2)
    
        if not conversation:
            return False
        
        observations = [turn["formatted"] for turn in conversation]
    
        if not observations:
            return False
        
        context_text = "\n".join(observations)
        system_prompt = get_profile_update_prompt(ctx.user_name)
        user_content = json.dumps({
            "entity_name": ctx.user_name,
            "entity_type": "person",
            "existing_summary": profile.get("summary", ""),
            "new_observations": context_text,
            "known_aliases": [ctx.user_name]
        }, indent=2)
        
        raw_response = await self.llm.call_reasoning(system_prompt, user_content)

        if not raw_response:
            return None

        new_summary = self._extract_summary(raw_response)
        
        if not new_summary or new_summary == profile.get("summary", ""):
            return None
        
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            partial(self.resolver.update_profile_summary, user_id, new_summary)
        )
        
        current_msg_id = await ctx.redis.get("global:next_msg_id")
        current_msg_id = int(current_msg_id) if current_msg_id else 0
        
        await loop.run_in_executor(
            self.executor,
            partial(
                self.store.update_entity_profile,
                entity_id=user_id,
                canonical_name=ctx.user_name,
                summary=new_summary,
                embedding=embedding,
                last_msg_id=current_msg_id,
                topic=profile.get("topic", "Personal")
            )
        )
        
        logger.info(f"Refined user profile for {ctx.user_name}")
        return True
    

    async def _run_updates(self, ctx: JobContext, entity_ids: List[int], conversation: List[Dict]):
        current_msg_id = await ctx.redis.get("global:next_msg_id")
        current_msg_id = int(current_msg_id) if current_msg_id else 0
        
        semaphore = asyncio.Semaphore(5)

        async def update_single(ent_id: int) -> Optional[dict]:
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
                
                observations = []

                for i, turn in enumerate(conversation):
                    if turn["role"] == "user" and pattern.search(turn["raw"]):
                        if i > 0 and conversation[i-1]["role"] == "assistant":
                            observations.append(conversation[i-1]["formatted"])
                        observations.append(turn["formatted"])
                        if i + 1 < len(conversation) and conversation[i+1]["role"] == "assistant":
                            observations.append(conversation[i+1]["formatted"])
            
                if not observations:
                    return None
                
                system_prompt = get_profile_update_prompt(ctx.user_name)
                user_content = json.dumps({
                    "entity_name": canonical_name,
                    "entity_type": entity_type,
                    "existing_summary": existing_summary,
                    "new_observations": observations,
                    "known_aliases": mentions
                }, indent=2)
                
                raw_response = await self.llm.call_reasoning(system_prompt, user_content)

                if not raw_response:
                    return None

                new_summary = self._extract_summary(raw_response)

                if not new_summary or new_summary == existing_summary:
                    return None
                
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    self.executor,
                    partial(self.resolver.update_profile_summary, ent_id, new_summary)
                )
                
                logger.info(f"Refined profile for {canonical_name} (ID: {ent_id})")
                
                return {
                    "id": ent_id,
                    "canonical_name": canonical_name,
                    "summary": new_summary,
                    "topic": profile.get("topic", "General"),
                    "embedding": embedding,
                    "last_msg_id": current_msg_id
                }
        
        results = await asyncio.gather(*[update_single(eid) for eid in entity_ids])
        return [r for r in results if r is not None]

    async def _write_updates(self, updates: List[dict]):
        """Write profile updates directly to Memgraph."""
        loop = asyncio.get_running_loop()
        
        for update in updates:
            await loop.run_in_executor(
                self.executor,
                partial(
                    self.store.update_entity_profile,
                    entity_id=update["id"],
                    canonical_name=update["canonical_name"],
                    summary=update["summary"],
                    embedding=update["embedding"],
                    last_msg_id=update["last_msg_id"],
                    topic=update["topic"]
                )
            )
        
        logger.info(f"Wrote {len(updates)} profile updates to graph")