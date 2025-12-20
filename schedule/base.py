from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import redis.asyncio as redis



@dataclass
class JobContext:
    """Context passed to every job method."""
    user_name: str
    redis: redis.Redis
    idle_seconds: float = 0.0
    last_run: Optional[datetime] = None


@dataclass 
class JobResult:
    """Result returned from job execution."""
    success: bool = True
    summary: str = ""
    reschedule_seconds: Optional[float] = None


class BaseJob(ABC):
    """Base class for scheduled jobs."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def should_run(self, ctx: JobContext) -> bool:
        pass
    
    @abstractmethod
    async def execute(self, ctx: JobContext) -> JobResult:
        pass
    
    async def on_shutdown(self, ctx: JobContext) -> None:
        """Override for cleanup. Default no-op."""
        pass