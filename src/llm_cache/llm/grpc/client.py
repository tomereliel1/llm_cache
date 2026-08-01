from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.llm.grpc.generated import llm_pb2, llm_pb2_grpc
from llm_cache.llm.interface import ILLMProvider
from llm_cache.request_context import grpc_metadata_for_current_request


class LLMGrpcClient(ILLMProvider):
    """Client adapter that exposes a remote LLM service as ``ILLMProvider``."""

    def __init__(
        self,
        target: str = "localhost:50053",
        timeout_seconds: float = 300.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        """Create an LLM gRPC client.

        Args:
            target: gRPC target address for the LLM service.
            timeout_seconds: Deadline used for each generation request.
            channel: Optional existing channel, mainly for tests. When omitted, this
                client creates and owns its channel.
        """
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = llm_pb2_grpc.LLMServiceStub(self._channel)

    def generate_answer(self, prompt: str) -> str:
        """Request an answer from the remote LLM service.

        Args:
            prompt: Prompt to send to the LLM provider.

        Returns:
            Generated answer text.

        Raises:
            ProviderUnavailableError: If the LLM service cannot be reached.
            ProviderTimeoutError: If the request exceeds the configured deadline.
            RuntimeError: If the service returns another gRPC error status.
        """
        request = llm_pb2.GenerateRequest(prompt=prompt)
        metadata = grpc_metadata_for_current_request()

        try:
            if metadata is None:
                reply = self._stub.Generate(request, timeout=self._timeout_seconds)
            else:
                reply = self._stub.Generate(
                    request,
                    timeout=self._timeout_seconds,
                    metadata=metadata,
                )
        except grpc.RpcError as error:
            code = error.code()
            if code is grpc.StatusCode.UNAVAILABLE:
                raise ProviderUnavailableError("LLM", self._target, error.details()) from error
            if code is grpc.StatusCode.DEADLINE_EXCEEDED:
                raise ProviderTimeoutError(
                    "LLM",
                    self._target,
                    error.details(),
                    self._timeout_seconds,
                ) from error
            code_name = code.name if code is not None else code
            raise RuntimeError(f"LLM gRPC call failed: {code_name}: {error.details()}") from error

        return reply.response

    def close(self) -> None:
        """Close the owned gRPC channel, if this client created one."""
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
