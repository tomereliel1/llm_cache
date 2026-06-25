from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.embedding import embedding_pb2, embedding_pb2_grpc
from llm_cache.embedding.i_embedder import IEmbedder


class EmbeddingGrpcClient(IEmbedder):
    """Client-side adapter that makes a remote embedding service look like IEmbedder."""

    def __init__(
        self,
        target: str = "localhost:50051",
        timeout_seconds: float = 30.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)

    def embed(self, prompt: str) -> list[float]:
        request = embedding_pb2.EmbedRequest(prompt=prompt)

        try:
            reply = self._stub.Embed(request, timeout=self._timeout_seconds)
        except grpc.RpcError as error:
            code = error.code()
            code_name = code.name if code is not None else code
            raise RuntimeError(
                f"Embedding gRPC call failed: {code_name}: {error.details()}"
            ) from error

        return list(reply.vector)

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
