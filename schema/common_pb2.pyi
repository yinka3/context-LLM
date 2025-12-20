from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    USER_MESSAGE: _ClassVar[MessageType]
    PROFILE_UPDATE: _ClassVar[MessageType]
    SYSTEM_ENTITY: _ClassVar[MessageType]
USER_MESSAGE: MessageType
PROFILE_UPDATE: MessageType
SYSTEM_ENTITY: MessageType

class Entity(_message.Message):
    __slots__ = ("id", "canonical_name", "type", "confidence", "summary", "topic", "embedding", "last_profiled_msg_id", "aliases")
    ID_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    LAST_PROFILED_MSG_ID_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    id: int
    canonical_name: str
    type: str
    confidence: float
    summary: str
    topic: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    last_profiled_msg_id: int
    aliases: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[int] = ..., canonical_name: _Optional[str] = ..., type: _Optional[str] = ..., confidence: _Optional[float] = ..., summary: _Optional[str] = ..., topic: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ..., last_profiled_msg_id: _Optional[int] = ..., aliases: _Optional[_Iterable[str]] = ...) -> None: ...

class Relationship(_message.Message):
    __slots__ = ("message_id", "entity_a", "entity_b", "confidence")
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_A_FIELD_NUMBER: _ClassVar[int]
    ENTITY_B_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    message_id: int
    entity_a: str
    entity_b: str
    confidence: float
    def __init__(self, message_id: _Optional[int] = ..., entity_a: _Optional[str] = ..., entity_b: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class BatchMessage(_message.Message):
    __slots__ = ("type", "list_ents", "list_relations")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    LIST_ENTS_FIELD_NUMBER: _ClassVar[int]
    LIST_RELATIONS_FIELD_NUMBER: _ClassVar[int]
    type: MessageType
    list_ents: _containers.RepeatedCompositeFieldContainer[Entity]
    list_relations: _containers.RepeatedCompositeFieldContainer[Relationship]
    def __init__(self, type: _Optional[_Union[MessageType, str]] = ..., list_ents: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., list_relations: _Optional[_Iterable[_Union[Relationship, _Mapping]]] = ...) -> None: ...
