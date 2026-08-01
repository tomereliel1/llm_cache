from __future__ import annotations

from collections.abc import Iterator

import grpc
import pytest

from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.vector_store import InMemoryVectorStore, VectorStoreGrpcClient
from llm_cache.vector_store.grpc.service import create_vector_store_grpc_server


class DeadlineExceededError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "deadline exceeded"


class UnavailableError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "connection refused"


class DeadlineExceededStub:
    def SearchSimilar(self, request, timeout: float):
        raise DeadlineExceededError()


class UnavailableStub:
    def SearchSimilar(self, request, timeout: float):
        raise UnavailableError()


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
        entry_id = client.store(
            prompt="What is semantic caching?",
            response="cached response",
            vector=[1.0, 0.0],
        )

        result = client.search_similar([0.9, 0.1])

    assert entry_id == "entry-1"
    assert result.found is True
    assert result.prompt == "What is semantic caching?"
    assert result.response == "cached response"
    assert result.score == pytest.approx(0.9938837)


def test_vector_store_grpc_client_returns_miss(
    vector_store_grpc_target: str,
) -> None:
    with VectorStoreGrpcClient(target=vector_store_grpc_target) as client:
        result = client.search_similar([1.0, 0.0])

    assert result.found is False
    assert result.prompt == ""
    assert result.response == ""


def test_vector_store_grpc_client_translates_grpc_errors() -> None:
    client = VectorStoreGrpcClient(target="localhost:1", timeout_seconds=0.1)
    client._stub = UnavailableStub()

    try:
        with pytest.raises(
            ProviderUnavailableError,
            match=r"^Vector store service is unavailable\.",
        ) as exc_info:
            client.search_similar([1.0, 0.0])
    finally:
        client.close()

    assert "Provider: Vector store" in exc_info.value.technical_details
    assert "Target: localhost:1" in exc_info.value.technical_details


def test_vector_store_grpc_client_translates_deadline_exceeded_error() -> None:
    client = VectorStoreGrpcClient(target="localhost:50052", timeout_seconds=0.5)
    client._stub = DeadlineExceededStub()

    try:
        with pytest.raises(
            ProviderTimeoutError,
            match=r"Increase the provider timeout using --provider-timeout-seconds",
        ) as exc_info:
            client.search_similar([1.0, 0.0])
    finally:
        client.close()

    assert "gRPC status: DEADLINE_EXCEEDED" in exc_info.value.technical_details
    assert "Timeout seconds: 0.5" in exc_info.value.technical_details
