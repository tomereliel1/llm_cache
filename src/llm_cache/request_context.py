from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

import grpc

REQUEST_ID_METADATA_KEY = "request-id"

_current_request_id: ContextVar[str | None] = ContextVar(
    "current_request_id",
    default=None,
)


def new_request_id() -> str:
    return str(uuid4())


def get_current_request_id() -> str | None:
    return _current_request_id.get()


@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    token = _current_request_id.set(request_id)
    try:
        yield
    finally:
        _current_request_id.reset(token)


def request_id_from_grpc_context(context: grpc.ServicerContext) -> str | None:
    metadata = dict(context.invocation_metadata())
    request_id = metadata.get(REQUEST_ID_METADATA_KEY)
    return request_id.strip() if request_id else None


def grpc_metadata_for_current_request() -> tuple[tuple[str, str], ...] | None:
    request_id = get_current_request_id()
    if request_id is None:
        return None

    return ((REQUEST_ID_METADATA_KEY, request_id),)
