from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SearchSimilarRequest(_message.Message):
    __slots__ = ("vector",)
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    vector: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vector: _Optional[_Iterable[float]] = ...) -> None: ...

class SearchSimilarReply(_message.Message):
    __slots__ = ("found", "prompt", "response", "similarity_score")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    found: bool
    prompt: str
    response: str
    similarity_score: float
    def __init__(self, found: _Optional[bool] = ..., prompt: _Optional[str] = ..., response: _Optional[str] = ..., similarity_score: _Optional[float] = ...) -> None: ...

class StoreRequest(_message.Message):
    __slots__ = ("prompt", "response", "vector")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    response: str
    vector: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, prompt: _Optional[str] = ..., response: _Optional[str] = ..., vector: _Optional[_Iterable[float]] = ...) -> None: ...

class StoreReply(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: _Optional[bool] = ...) -> None: ...
