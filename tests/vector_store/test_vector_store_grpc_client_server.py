from __future__ import annotations

from collections.abc import Iterator

import pytest

from llm_cache.vector_store import InMemoryVectorStore, VectorStoreGrpcClient
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server


@pytest.fixture
def vector_store_grpc_target() -> Iterator[str]:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    server = create_vector_store_grpc_server(vector_store, max_workers=2)
    port = server.add_insecure_port("localhost:0")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        server.stop(grace=0)


def test_vector_store_grpc_client_stores_and_searches(
    vector_store_grpc_target: str,
) -> None:
    with VectorStoreGrpcClient(target=vector_store_grpc_target) as client:
        client.store(
            prompt="What is semantic caching?",
            response="cached response",
            vector=[1.0, 0.0],
        )

        result = client.search_similar([0.9, 0.1])

    assert result.found is True
    assert result.prompt == "What is semantic caching?"
    assert result.response == "cached response"


def test_vector_store_grpc_client_returns_miss(
    vector_store_grpc_target: str,
) -> None:
    with VectorStoreGrpcClient(target=vector_store_grpc_target) as client:
        result = client.search_similar([1.0, 0.0])

    assert result.found is False
    assert result.prompt == ""
    assert result.response == ""


def test_vector_store_grpc_client_translates_grpc_errors() -> None:
    with VectorStoreGrpcClient(target="localhost:1", timeout_seconds=0.1) as client:
        with pytest.raises(RuntimeError, match="Vector store search gRPC call failed"):
            client.search_similar([1.0, 0.0])
