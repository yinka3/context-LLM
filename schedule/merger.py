import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import numpy as np
from loguru import logger

from schedule.base import BaseJob, JobContext, JobResult

if TYPE_CHECKING:
    from main.entity_resolve import EntityResolver
    from graph.memgraph import MemGraphStore


class MergeDetectionJob(BaseJob):
    """
    Detects and merges duplicate entities based on embedding similarity.
    
    Trigger: User idle for IDLE_THRESHOLD seconds, hasn't run this session.
    Auto-merges high-confidence pairs (>= 0.9), stores others for HITL review.
    """
    
    IDLE_THRESHOLD = 15 * 60
    AUTO_MERGE_THRESHOLD = 0.9
    HITL_THRESHOLD = 0.65
    
    def __init__(self, ent_resolver: "EntityResolver", store: "MemGraphStore"):
        self.ent_resolver = ent_resolver
        self.store = store
    
    @property
    def name(self) -> str:
        return "merge_detection"
    
    async def should_run(self, ctx: JobContext) -> bool:
        if ctx.idle_seconds < self.IDLE_THRESHOLD:
            return False
        
        ran_key = f"merge_ran:{ctx.user_name}"
        already_ran = await ctx.redis.get(ran_key)
        return not already_ran
    
    async def execute(self, ctx: JobContext) -> JobResult:
        await ctx.redis.set(f"merge_ran:{ctx.user_name}", "true")
        
        loop = asyncio.get_running_loop()
        
        candidates = await loop.run_in_executor(
            None, self.ent_resolver.detect_merge_candidates
        )
        
        if not candidates:
            return JobResult(success=True, summary="No merge candidates found")
        
        auto_merge = [c for c in candidates if c["cross_score"] >= self.AUTO_MERGE_THRESHOLD]
        hitl = [c for c in candidates if self.HITL_THRESHOLD <= c["cross_score"] < self.AUTO_MERGE_THRESHOLD]
        
        logger.info(f"Merge split: {len(auto_merge)} auto, {len(hitl)} HITL")
        
        merged_ids = set()
        successful = 0
        failed = 0
        
        for candidate in auto_merge:
            primary_id = candidate["primary_id"]
            secondary_id = candidate["secondary_id"]
            
            if primary_id in merged_ids or secondary_id in merged_ids:
                continue
            
            success = await self._execute_merge(primary_id, secondary_id)
            
            if success:
                merged_ids.add(secondary_id)
                successful += 1
                self._sync_resolver(primary_id, secondary_id)
            else:
                failed += 1
        
        proposals_stored = await self._store_hitl_proposals(ctx, hitl, merged_ids)
        
        if successful > 0:
            mentions = self.ent_resolver.get_mentions()
            await ctx.redis.hset("entity_mentions", mapping=mentions)
        
        return JobResult(
            success=True,
            summary=f"{successful} merged, {failed} failed, {proposals_stored} HITL proposals"
        )
    
    async def on_shutdown(self, ctx: JobContext) -> None:
        """Set pending flag so next session picks up merge work."""
        await ctx.redis.set(f"pending:{ctx.user_name}:{self.name}", "true")
        logger.debug("Merge detection pending flag set")
    
    async def _execute_merge(self, primary_id: int, secondary_id: int, max_retries: int = 2) -> bool:
        """Execute merge with retry."""
        loop = asyncio.get_running_loop()
        
        primary_profile = self.ent_resolver.entity_profiles.get(primary_id, {})
        secondary_profile = self.ent_resolver.entity_profiles.get(secondary_id, {})
        merged_summary = self._merge_summaries(
            primary_profile.get("summary", ""),
            secondary_profile.get("summary", "")
        )
        
        for attempt in range(1, max_retries + 1):
            success = await loop.run_in_executor(
                None,
                self.store.merge_entities,
                primary_id,
                secondary_id,
                merged_summary
            )
            
            if success:
                return True
            
            logger.warning(f"Merge attempt {attempt}/{max_retries} failed")
            if attempt < max_retries:
                await asyncio.sleep(1.0 * attempt)
        
        return False
    
    def _merge_summaries(self, primary: str, secondary: str) -> str:
        """Combine two summaries. TODO: LLM-based merging."""
        if not primary:
            return secondary
        if not secondary:
            return primary
        return f"{primary} {secondary}"
    
    def _sync_resolver(self, primary_id: int, secondary_id: int):
        """Update EntityResolver after merge."""
        secondary_aliases = self.ent_resolver.get_mentions_for_id(secondary_id)
        
        with self.ent_resolver._lock:
            for alias in secondary_aliases:
                self.ent_resolver._name_to_id[alias] = primary_id
            
            if secondary_id in self.ent_resolver.entity_profiles:
                del self.ent_resolver.entity_profiles[secondary_id]
        
        try:
            self.ent_resolver.index_id_map.remove_ids(
                np.array([secondary_id], dtype=np.int64)
            )
        except Exception as e:
            logger.warning(f"FAISS removal failed for {secondary_id}: {e}")
    
    async def _store_hitl_proposals(self, ctx: JobContext, proposals: list, merged_ids: set) -> int:
        """Store merge proposals for human review."""
        stored = 0
        proposal_key = f"merge_proposals:{ctx.user_name}"
        
        for candidate in proposals:
            primary_id = candidate["primary_id"]
            secondary_id = candidate["secondary_id"]
            
            if primary_id in merged_ids or secondary_id in merged_ids:
                continue
            
            proposal = {
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "primary_name": candidate["primary_name"],
                "secondary_name": candidate["secondary_name"],
                "faiss_score": candidate["faiss_score"],
                "cross_score": candidate["cross_score"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending"
            }
            
            await ctx.redis.rpush(proposal_key, json.dumps(proposal))
            stored += 1
        
        return stored