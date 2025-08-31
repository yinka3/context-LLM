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
    

@dataclass(frozen=True)
class AttributeData:
    value: Union[str, 'EntityData']
    message: MessageData
    confidence_score: float
    mentioned_in: List[Tuple] = field(default_factory=list)

@dataclass
class EntityData:
    id: int
    name: str
    type: str
    aliases: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    attributes: Dict[str, List[AttributeData]] = field(default_factory=dict)
    mentioned_in: List[int] = field(default_factory=list)


@dataclass
class EdgeData:
    type: str
    source_messages_id: List[int]
    confidence_score: float

@dataclass
class Tier2Task:
    task_id: str
    message_id: str
    trigger_reason: str
    priority: int
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now())
