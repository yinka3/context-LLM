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
    

@dataclass
class EntityData:
    id: int
    name: str
    type: str
    aliases: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    mentioned_in: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class BridgeData:
    type: str
    confidence: float
    value: Union[EntityData, str]

@dataclass(frozen=True)
class EdgeData:
    messages_connected: List[int]
    bridge: BridgeData
