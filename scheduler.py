import asyncio
from datetime import datetime, timezone
from loguru import logger
from typing import TYPE_CHECKING, Optional
from redisclient import AsyncRedisClient

if TYPE_CHECKING:
    from main.entity_resolve import EntityResolver
    from graph.memgraph import MemGraphStore

class Scheduler:

    INACTIVE_CHECK = 60
    INACTIVE_T = 15 * 60
    PROFILE_STALE_T = 24 * 60 * 60

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
        await self._check_pending_profile()

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
        await self.redis.set(f"last_message_at:{self.user_name}"), datetime.now(timezone.utc).isoformat()
        await self.redis.delete(f"merge_ran:{self.user_name}")
    
    async def _inactive_loop(self):
        while self._is_running:
            await asyncio.sleep(self.INACTIVITY_CHECK_INTERVAL)
            
            last_activity = await self.redis.get(f"last_message_at:{self.user_name}")
            if not last_activity:
                continue
            
            last_ts = datetime.fromisoformat(last_activity.decode())
            idle_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
            
            if idle_seconds >= self.INACTIVITY_THRESHOLD:
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

    async def _check_pending_profile(self):
        last_run = await self.redis.get(f"last_profile_run:{self.user_name}")
        
        if last_run is None:
            return
        
        last_ts = datetime.fromisoformat(last_run.decode())
        staleness = (datetime.now(timezone.utc) - last_ts).total_seconds()
        
        if staleness > self.PROFILE_STALE_THRESHOLD:
            logger.info(f"Profile refinement due (stale by {staleness/3600:.1f}h)")
            await self._run_profile_refinement()
            await self.redis.set(
                f"last_profile_run:{self.user_name}",
                datetime.now(timezone.utc).isoformat()
        )


    async def run_merge_job(self):
        logger.info("Starting merge job...")
        await self.redis.set(f"merge_ran:{self.user_name}", "true")
        pass

    async def _run_profile_refinement(self):
        logger.info("Running profile refinement...")
        pass