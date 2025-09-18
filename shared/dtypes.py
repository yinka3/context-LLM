from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class MessageData:
    id: int
    role: str
    message: str
    sentiment: str
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    bridges: List['BridgeData'] = field(default_factory=list)
    

@dataclass
class EntityData:
    id: int
    name: str
    type: str
    aliases: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    mentioned_in: List[int] = field(default_factory=list)
    contextual_mentions: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, EntityData):
            return NotImplemented
        return self.id == other.id

@dataclass(frozen=True)
class BridgeData:
    type: str
    value: Union[EntityData, str]

@dataclass(frozen=True)
class EdgeData:
    messages: Tuple[int, int]
    bridge: BridgeData

@dataclass
class Tier2Task:
    task_id: str
    message_id: str
    trigger_reason: str
    priority: int
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now())
