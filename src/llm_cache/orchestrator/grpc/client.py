from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.orchestrator.errors import OrchestratorClientError
from llm_cache.orchestrator.grpc.generated import orchestrator_pb2, orchestrator_pb2_grpc
from llm_cache.orchestrator.orchestrator import QueryResult
from llm_cache.request_context import (
    get_current_request_id,
    grpc_metadata_for_current_request,
    new_request_id,
    request_context,
)


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
        if get_current_request_id() is None:
            with request_context(new_request_id()):
                return self.query(prompt)

        try:
            reply = self._stub.SubmitPrompt(
                orchestrator_pb2.SubmitPromptRequest(prompt=prompt),
                timeout=self._timeout_seconds,
                metadata=grpc_metadata_for_current_request(),
            )
        except grpc.RpcError as error:
            code = error.code()
            if code is grpc.StatusCode.UNAVAILABLE:
                details = error.details()
                metadata = dict(error.trailing_metadata() or ())
                technical_details = metadata.get("technical-details-bin")
                if isinstance(technical_details, bytes):
                    technical_details = technical_details.decode("utf-8", errors="replace")
                if technical_details:
                    raise OrchestratorClientError(
                        details, technical_details=technical_details
                    ) from error
                raise OrchestratorClientError(
                    "Orchestrator service is unavailable. Please try again later.",
                    technical_details=f"gRPC status: UNAVAILABLE\nDetails: {details}",
                ) from error
            code_name = code.name if code is not None else code
            raise RuntimeError(
                f"Orchestrator gRPC call failed: {code_name}: {error.details()}"
            ) from error

        return QueryResult(response=reply.response, cache_hit=reply.cache_hit)

    def is_ready(self, timeout_seconds: float = 1.0) -> bool:
        """Return whether the orchestrator channel becomes ready before the timeout."""
        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout_seconds)
        except grpc.FutureTimeoutError:
            return False
        return True

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
