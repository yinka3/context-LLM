from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class MessageData(BaseModel):
    id: int = -1
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class EntityPair(BaseModel):
    entity_a: str = Field(..., description="First entity canonical_name (alphabetically first).")
    entity_b: str = Field(..., description="Second entity canonical_name (alphabetically second).")
    confidence: float = Field(..., description="0.0 to 1.0")

class MessageConnections(BaseModel):
    message_id: int = Field(..., description="Copy exactly from input.")
    entity_pairs: List[EntityPair] = Field(default_factory=list, description="Pairs of entities with meaningful connections in this message.")

class ConnectionExtractionResponse(BaseModel):
    message_results: List[MessageConnections] = Field(..., description="Per-message entity connections.")

class ProfileUpdate(BaseModel):
    canonical_name: str
    summary: str = Field(..., description="Updated biographical summary merging old facts with new observations.")
    topic: str = Field(..., description="Broad thematic category.")

class AmbiguousResolution(BaseModel):
    mention: str = Field(..., description="The original mention text being resolved.")
    resolved_id: Optional[int] = Field(None, description="ID of matched existing entity. Null if is_new=True.")
    canonical_name: Optional[str] = Field(None, description="Canonical name of matched entity.")
    is_new: bool = Field(False, description="True if this mention doesn't match any candidate.")

class NewEntityGroup(BaseModel):
    canonical_name: str = Field(..., description="The most complete/formal name for this entity.")
    type: str = Field(..., description="Entity type: PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, TOPIC.")
    mentions: List[str] = Field(..., description="All mention texts that refer to this same entity.")

class DisambiguationResponse(BaseModel):
    ambiguous_resolutions: List[AmbiguousResolution] = Field(default_factory=list, description="Resolution decisions for ambiguous mentions.")
    new_entity_groups: List[NewEntityGroup] = Field(default_factory=list, description="Grouped new mentions that refer to the same entity.")