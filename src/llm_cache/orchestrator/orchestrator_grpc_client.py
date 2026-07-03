from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.orchestrator import orchestrator_pb2, orchestrator_pb2_grpc
from llm_cache.orchestrator.cache_orchestrator import QueryResult


class OrchestratorGrpcClient:
    """Client adapter for the public orchestrator gRPC service."""

    def __init__(
        self,
        target: str = "localhost:50050",
        timeout_seconds: float = 30.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = orchestrator_pb2_grpc.OrchestratorServiceStub(self._channel)

    def query(self, prompt: str) -> QueryResult:
        try:
            reply = self._stub.SubmitPrompt(
                orchestrator_pb2.SubmitPromptRequest(prompt=prompt),
                timeout=self._timeout_seconds,
            )
        except grpc.RpcError as error:
            code = error.code()
            code_name = code.name if code is not None else code
            raise RuntimeError(
                f"Orchestrator gRPC call failed: {code_name}: {error.details()}"
            ) from error

        return QueryResult(response=reply.response, cache_hit=reply.cache_hit)

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
