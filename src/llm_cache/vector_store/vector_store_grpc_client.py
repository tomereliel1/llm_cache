from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.vector_store import vector_store_pb2, vector_store_pb2_grpc
from llm_cache.vector_store.i_vector_store import IVectorStore, VectorStoreResult


class VectorStoreGrpcClient(IVectorStore):
    """Client-side adapter that makes a remote vector store service look like IVectorStore."""

    def __init__(
        self,
        target: str = "localhost:50052",
        timeout_seconds: float = 30.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = vector_store_pb2_grpc.VectorStoreServiceStub(self._channel)

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        request = vector_store_pb2.SearchSimilarRequest(vector=vector)

        try:
            reply = self._stub.SearchSimilar(request, timeout=self._timeout_seconds)
        except grpc.RpcError as error:
            raise self._grpc_error("Vector store search gRPC call failed", error) from error

        return VectorStoreResult(
            found=reply.found,
            prompt=reply.prompt,
            response=reply.response,
        )

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        request = vector_store_pb2.StoreRequest(
            prompt=prompt,
            response=response,
            vector=vector,
        )

        try:
            reply = self._stub.Store(request, timeout=self._timeout_seconds)
        except grpc.RpcError as error:
            raise self._grpc_error("Vector store store gRPC call failed", error) from error

        if not reply.success:
            raise RuntimeError("Vector store gRPC store call did not succeed")

        return ""

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

    @staticmethod
    def _grpc_error(message: str, error: grpc.RpcError) -> RuntimeError:
        code = error.code()
        code_name = code.name if code is not None else code
        return RuntimeError(f"{message}: {code_name}: {error.details()}")
