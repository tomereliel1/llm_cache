from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures

import grpc
import pytest

from llm_cache.embedding import embedding_pb2_grpc
from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.embedding.embedding_grpc_service import EmbeddingGrpcService
from llm_cache.test_doubles import EmbedderStub


@pytest.fixture
def embedding_grpc_target() -> Iterator[str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    embedder = EmbedderStub(vector=[0.1, 0.2, 0.3])
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGrpcService(embedder),
        server,
    )

    port = server.add_insecure_port("localhost:0")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        server.stop(grace=0)


def test_embedding_grpc_client_returns_vector(embedding_grpc_target: str) -> None:
    with EmbeddingGrpcClient(target=embedding_grpc_target) as client:
        assert client.embed("hello") == pytest.approx([0.1, 0.2, 0.3])


def test_embedding_grpc_client_translates_grpc_errors() -> None:
    with EmbeddingGrpcClient(target="localhost:1", timeout_seconds=0.1) as client:
        with pytest.raises(RuntimeError, match="Embedding gRPC call failed"):
            client.embed("hello")
