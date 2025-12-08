from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOC"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"

class MessageData(BaseModel):
    id: int = -1
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ExtractedRelationship(BaseModel):
    source: str = Field(..., description="Canonical name of the source entity.")
    target: str = Field(..., description="Canonical name of the target entity.")
    relation: str = Field(..., description="A short, active verb (e.g., 'owns', 'dislikes').")
    confidence: float

class EntityMention(BaseModel):
    canonical_name: str = Field(..., description="The deduplicated entity name from the entities list.")
    original_text: Optional[str] = Field(None, description="The meaningful alias/reference used in the message. Only include if different from canonical_name and useful as an alias (e.g., 'my sister', 'the CEO'). Omit for pronouns, exact name matches, or noisy references.")

class MessageExtraction(BaseModel):
    message_id: int = Field(..., description="Copy exactly from input. Do not modify or generate.")
    entity_mentions: List[EntityMention] = Field(..., description="Entities mentioned in this specific message with optional alias capture.")
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class ProfileUpdate(BaseModel):
    canonical_name: str
    aliases: List[str]
    type: str = Field(..., description="Entity type (PERSON, ORG, LOC, etc.)")
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

class RelationshipExtractionResponse(BaseModel):
    message_extractions: List[MessageExtraction] = Field(..., description="Per-message entity mentions and relationships.")
    entities_with_new_info: List[int] = Field(default_factory=list, description="Entity IDs that have new biographical information in these messages.")