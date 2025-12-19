from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional


class MessageData(BaseModel):
    id: int = -1
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class EntityPair(BaseModel):
    entity_a: str = Field(..., description="First entity canonical_name (alphabetically first).")
    entity_b: str = Field(..., description="Second entity canonical_name (alphabetically second).")
    entity_b: str = Field(..., description="The second entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reason: str = Field(..., description="Short explanation for the connection")

class MessageConnections(BaseModel):
    message_id: int = Field(..., description="Copy exactly from input.")
    entity_pairs: List[EntityPair] = Field(default_factory=list, description="Pairs of entities with meaningful connections in this message.")

class ConnectionExtractionResponse(BaseModel):
    message_results: List[MessageConnections] = Field(..., description="Per-message entity connections.")
    reasoning_trace: str = Field(..., description="Chain of thought analysis before extraction")

class ProfileUpdate(BaseModel):
    canonical_name: str
    summary: str = Field(..., description="Updated biographical summary merging old facts with new observations.")
    topic: str = Field(..., description="Broad thematic category.")

class ResolutionEntry(BaseModel):
    verdict: Literal["EXISTING", "NEW_GROUP", "NEW_SINGLE"]
    mentions: List[str]
    entity_type: str
    canonical_name: Optional[str] = None

class DisambiguationResult(BaseModel):
    entries: List[ResolutionEntry]