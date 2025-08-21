from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Union

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
    mentioned_in: List[int] = field(default_factory=list)

@dataclass(frozen=True)
class EntityData:
    id: int
    name: str
    type: str
    attributes: Dict[str, List[AttributeData]] = field(default_factory=dict)


@dataclass
class EdgeData:
    type: str
    source_messages_id: List[int]
    confidence_score: float
