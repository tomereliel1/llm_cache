from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures

import grpc
import pytest

from llm_cache.embedding.grpc.client import EmbeddingGrpcClient
from llm_cache.embedding.grpc.generated import embedding_pb2_grpc
from llm_cache.embedding.grpc.service import EmbeddingGrpcService
from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.test_doubles import EmbedderStub


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
    def Embed(self, request, timeout: float):
        raise DeadlineExceededError()


class UnavailableStub:
    def Embed(self, request, timeout: float):
        raise UnavailableError()


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
    client = EmbeddingGrpcClient(target="localhost:1", timeout_seconds=0.1)
    client._stub = UnavailableStub()

    try:
        with pytest.raises(
            ProviderUnavailableError,
            match=r"^Embedding service is unavailable\.",
        ) as exc_info:
            client.embed("hello")
    finally:
        client.close()

    assert "Provider: Embedding" in exc_info.value.technical_details
    assert "Target: localhost:1" in exc_info.value.technical_details


def test_embedding_grpc_client_translates_deadline_exceeded_error() -> None:
    client = EmbeddingGrpcClient(target="localhost:50051", timeout_seconds=0.5)
    client._stub = DeadlineExceededStub()

    try:
        with pytest.raises(
            ProviderTimeoutError,
            match=r"Increase the provider timeout using --provider-timeout-seconds",
        ) as exc_info:
            client.embed("hello")
    finally:
        client.close()

    assert "gRPC status: DEADLINE_EXCEEDED" in exc_info.value.technical_details
    assert "Timeout seconds: 0.5" in exc_info.value.technical_details
