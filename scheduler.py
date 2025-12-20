import asyncio
from datetime import datetime, timezone
from enum import Enum
from loguru import logger
from typing import TYPE_CHECKING, Dict, Optional
from redisclient import AsyncRedisClient
import numpy as np

if TYPE_CHECKING:
    from main.entity_resolve import EntityResolver
    from graph.memgraph import MemGraphStore


class Scheduler:

    INACTIVE_CHECK = 60
    INACTIVE_T = 15 * 60
    AUTO_MERGE_THRESHOLD = 0.9
    HITL_THRESHOLD = 0.65

    def __init__(self, user_name: str, entity_resolver: 'EntityResolver', graph_store: 'MemGraphStore'):

        self.user_name = user_name
        self.redis = AsyncRedisClient().get_client()
        self.ent_resolver = entity_resolver
        self.store = graph_store

        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self):
        self._is_running = True
        await self._check_pending_merge()
        self._monitor_task = asyncio.create_task(self._inactive_loop())
        logger.info(f"Scheduler started for {self.user_name}")


    async def stop(self):

        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.set(f"pending_merge:{self.user_name}", "true")
        logger.info("Scheduler stopped, pending_merge flag set")
    
    async def record_activity(self):
        await self.redis.set(f"last_message_at:{self.user_name}", datetime.now(timezone.utc).isoformat())
        await self.redis.delete(f"merge_ran:{self.user_name}")
    
    async def _inactive_loop(self):
        while self._is_running:
            await asyncio.sleep(self.INACTIVE_CHECK)
            
            last_activity = await self.redis.get(f"last_message_at:{self.user_name}")
            if not last_activity:
                continue
            
            last_ts = datetime.fromisoformat(last_activity.decode())
            idle_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
            
            if idle_seconds >= self.INACTIVE_T:
                already_ran = await self.redis.get(f"merge_ran:{self.user_name}")
                if not already_ran:
                    logger.info(f"Inactivity threshold reached ({idle_seconds:.0f}s)")
                    await self.run_merge_job()
    
    async def _check_pending_merge(self):
        pending = await self.redis.get(f"pending_merge:{self.user_name}")
        
        if pending:
            logger.info("Pending merge from previous session detected")
            await self.redis.delete(f"pending_merge:{self.user_name}")
            await self.run_merge_job()

    async def run_merge_job(self):
        logger.info("Starting merge job...")
        await self.redis.set(f"merge_ran:{self.user_name}", "true")
        loop = asyncio.get_event_loop()

        candidates = await loop.run_in_executor(
            None, self.ent_resolver.detect_merge_candidates
        )

        if not candidates:
            logger.info("No merge candidate found")
            return

        logger.info(f"Found {len(candidates)} merge candidates")

        auto_merge = []
        hitl_proposals = []
        
        for candidate in candidates:
            score = candidate["cross_score"]
            if score >= self.AUTO_MERGE_THRESHOLD:
                auto_merge.append(candidate)
            elif score >= self.HITL_THRESHOLD:
                hitl_proposals.append(candidate)
        
        logger.info(f"Split: {len(auto_merge)} auto, {len(hitl_proposals)} HITL")
    
        merged_ids: set[int] = set()
        successful = 0
        failed = 0
        
        for candidate in auto_merge:
            primary_id = candidate["primary_id"]
            secondary_id = candidate["secondary_id"]
            
            if primary_id in merged_ids or secondary_id in merged_ids:
                continue
            
            success = await self._execute_merge_with_retry(
                primary_id, secondary_id, max_retries=2
            )
            
            if success:
                merged_ids.add(secondary_id)
                successful += 1
                self._sync_resolver_after_merge(primary_id, secondary_id)
            else:
                failed += 1
        
        proposals_stored = await self._store_hitl_proposals(hitl_proposals, merged_ids)
    
        if successful > 0:
            mentions = self.ent_resolver.get_mentions()
            await self.redis.hset("entity_mentions", mapping=mentions)
        
        logger.info(
            f"Merge job complete: {successful} auto-merged, "
            f"{failed} failed, {proposals_stored} proposals stored")
        
    async def _execute_merge_with_retry(
        self, primary_id: int, secondary_id: int, max_retries: int = 2) -> bool:
        """Execute merge with retry and backoff."""
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


    def _merge_summaries(self, primary_summary: str, secondary_summary: str) -> str:
        """
        Merge two entity summaries.
        TODO: Replace with LLM call for intelligent merging.
        """
        if not primary_summary:
            return secondary_summary
        if not secondary_summary:
            return primary_summary
        return f"{primary_summary} {secondary_summary}"


    def _sync_resolver_after_merge(self, primary_id: int, secondary_id: int):
        """Update EntityResolver state after merge."""
        import numpy as np
        
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
    
    async def _store_hitl_proposals(
        self, proposals: list[dict], merged_ids: set[int]) -> int:
        """Store merge proposals for human review."""
        import json
        from datetime import datetime, timezone
        
        stored = 0
        proposal_key = f"merge_proposals:{self.user_name}"
        
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
            
            await self.redis.rpush(proposal_key, json.dumps(proposal))
            stored += 1
            
            logger.debug(
                f"Stored HITL proposal: {candidate['primary_name']} <- "
                f"{candidate['secondary_name']} (score={candidate['cross_score']:.3f})"
            )
        
        return stored
