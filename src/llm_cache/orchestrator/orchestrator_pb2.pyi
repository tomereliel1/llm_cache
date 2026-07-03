from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SubmitPromptRequest(_message.Message):
    __slots__ = ("prompt",)
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    def __init__(self, prompt: _Optional[str] = ...) -> None: ...

class SubmitPromptReply(_message.Message):
    __slots__ = ("response", "cache_hit")
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
    response: str
    cache_hit: bool
    def __init__(self, response: _Optional[str] = ..., cache_hit: _Optional[bool] = ...) -> None: ...
