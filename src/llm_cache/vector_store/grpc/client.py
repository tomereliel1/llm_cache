from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.request_context import grpc_metadata_for_current_request
from llm_cache.vector_store.grpc.generated import vector_store_pb2, vector_store_pb2_grpc
from llm_cache.vector_store.interface import IVectorStore, VectorStoreResult


class VectorStoreGrpcClient(IVectorStore):
    """Client adapter that exposes a remote vector-store service as ``IVectorStore``."""

    def __init__(
        self,
        target: str = "localhost:50052",
        timeout_seconds: float = 30.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        """Create a vector-store gRPC client.

        Args:
            target: gRPC target address for the vector-store service.
            timeout_seconds: Deadline used for search and store requests.
            channel: Optional existing channel, mainly for tests. When omitted, this
                client creates and owns its channel.
        """
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = vector_store_pb2_grpc.VectorStoreServiceStub(self._channel)

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        """Search for a similar cached entry through the remote vector store.

        Args:
            vector: Embedding vector to search for.

        Returns:
            VectorStoreResult reported by the vector-store service.

        Raises:
            ProviderUnavailableError: If the vector-store service cannot be reached.
            ProviderTimeoutError: If the request exceeds the configured deadline.
            RuntimeError: If the service returns another gRPC error status.
        """
        request = vector_store_pb2.SearchSimilarRequest(vector=vector)
        metadata = grpc_metadata_for_current_request()

        try:
            if metadata is None:
                reply = self._stub.SearchSimilar(request, timeout=self._timeout_seconds)
            else:
                reply = self._stub.SearchSimilar(
                    request,
                    timeout=self._timeout_seconds,
                    metadata=metadata,
                )
        except grpc.RpcError as error:
            raise self._grpc_error("Vector store search gRPC call failed", error) from error

        return VectorStoreResult(
            found=reply.found,
            prompt=reply.prompt,
            response=reply.response,
            score=reply.similarity_score if reply.found else None,
        )

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        """Store a cache entry through the remote vector store.

        Args:
            prompt: Prompt text associated with the response.
            response: Response text to cache.
            vector: Embedding vector for the prompt.

        Returns:
            Entry identifier returned by the vector-store service.

        Raises:
            ProviderUnavailableError: If the vector-store service cannot be reached.
            ProviderTimeoutError: If the request exceeds the configured deadline.
            RuntimeError: If the service returns an error status or unsuccessful reply.
        """
        request = vector_store_pb2.StoreRequest(
            prompt=prompt,
            response=response,
            vector=vector,
        )
        metadata = grpc_metadata_for_current_request()

        try:
            if metadata is None:
                reply = self._stub.Store(request, timeout=self._timeout_seconds)
            else:
                reply = self._stub.Store(
                    request,
                    timeout=self._timeout_seconds,
                    metadata=metadata,
                )
        except grpc.RpcError as error:
            raise self._grpc_error("Vector store store gRPC call failed", error) from error

        if not reply.success:
            raise RuntimeError("Vector store gRPC store call did not succeed")

        return reply.entry_id

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

    def _grpc_error(self, message: str, error: grpc.RpcError) -> RuntimeError:
        """Translate vector-store gRPC failures into project-level exceptions.

        Args:
            message: Prefix used for generic runtime failures.
            error: gRPC error raised by the generated stub.

        Returns:
            Exception instance describing the provider failure.
        """
        code = error.code()
        if code is grpc.StatusCode.UNAVAILABLE:
            return ProviderUnavailableError("Vector store", self._target, error.details())
        if code is grpc.StatusCode.DEADLINE_EXCEEDED:
            return ProviderTimeoutError(
                "Vector store",
                self._target,
                error.details(),
                self._timeout_seconds,
            )
        code_name = code.name if code is not None else code
        return RuntimeError(f"{message}: {code_name}: {error.details()}")
