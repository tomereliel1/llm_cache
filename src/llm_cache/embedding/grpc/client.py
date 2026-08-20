from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.embedding.grpc.generated import embedding_pb2, embedding_pb2_grpc
from llm_cache.embedding.interface import IEmbedder
from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.request_context import grpc_metadata_for_current_request


class EmbeddingGrpcClient(IEmbedder):
    """Client adapter that exposes a remote embedding service as ``IEmbedder``."""

    def __init__(
        self,
        target: str = "localhost:50051",
        timeout_seconds: float = 30.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        """Create an embedding gRPC client.

        Args:
            target: gRPC target address for the embedding service.
            timeout_seconds: Deadline used for each embedding request.
            channel: Optional existing channel, mainly for tests. When omitted, this
                client creates and owns its channel.
        """
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)

    def embed(self, prompt: str) -> list[float]:
        """Request an embedding vector from the remote embedding service.

        Args:
            prompt: Text prompt to embed.

        Returns:
            Embedding vector returned by the service.

        Raises:
            ProviderUnavailableError: If the embedding service cannot be reached.
            ProviderTimeoutError: If the request exceeds the configured deadline.
            RuntimeError: If the service returns another gRPC error status.
        """
        request = embedding_pb2.EmbedRequest(prompt=prompt)
        metadata = grpc_metadata_for_current_request()

        try:
            if metadata is None:
                reply = self._stub.Embed(request, timeout=self._timeout_seconds)
            else:
                reply = self._stub.Embed(
                    request,
                    timeout=self._timeout_seconds,
                    metadata=metadata,
                )
        except grpc.RpcError as error:
            code = error.code()
            if code is grpc.StatusCode.UNAVAILABLE:
                raise ProviderUnavailableError(
                    "Embedding", self._target, error.details()
                ) from error
            if code is grpc.StatusCode.DEADLINE_EXCEEDED:
                raise ProviderTimeoutError(
                    "Embedding",
                    self._target,
                    error.details(),
                    self._timeout_seconds,
                ) from error
            code_name = code.name if code is not None else code
            raise RuntimeError(
                f"Embedding gRPC call failed: {code_name}: {error.details()}"
            ) from error

        return list(reply.vector)

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
