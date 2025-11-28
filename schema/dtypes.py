from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOC"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TOPIC = "TOPIC"

class MessageData(BaseModel):
    id: int
    role: str
    message: str
    sentiment: str
    timestamp: datetime = Field(default_factory=datetime.now)
    

class ExtractedEntity(BaseModel):
    id: Optional[int] = Field(None, description="The Integer ID of the matching candidate. MUST be null if this is a new entity.")
    canonical_name: str = Field(..., description="The most precise, capitalized name of the entity.")
    type: str = Field(..., description="The generic classification (PERSON, ORG, LOC, TOPIC, EVENT).")
    topic: str = Field(..., description="User-defined topic category this entity belongs to.")
    confidence: float = Field(..., description="Certainty score between 0.0 and 1.0.")
    has_new_info: bool = Field(False, description="True if the message contains NEW biographical facts about this entity.")

class ExtractedRelationship(BaseModel):
    source: str = Field(..., description="Canonical name of the source entity.")
    target: str = Field(..., description="Canonical name of the target entity.")
    relation: str = Field(..., description="A short, active verb (e.g., 'owns', 'dislikes').")
    confidence: float

class ExtractionResponse(BaseModel):
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]

class ProfileUpdate(BaseModel):
    canonical_name: str
    aliases: List[str]
    type: str = Field(..., description="Entity type (PERSON, ORG, LOC, etc.)")
    summary: str = Field(..., description="Updated biographical summary merging old facts with new observations.")
    topic: str = Field(..., description="Broad thematic category.")