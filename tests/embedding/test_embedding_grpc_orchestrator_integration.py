from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures

import grpc
import pytest

from llm_cache.embedding import embedding_pb2_grpc
from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.embedding.embedding_grpc_service import EmbeddingGrpcService
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy, VectorStoreMissStub


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


def test_orchestrator_works_with_embedding_grpc_client(
    embedding_grpc_target: str,
) -> None:
    with EmbeddingGrpcClient(target=embedding_grpc_target) as embedder:
        vector_store = VectorStoreMissStub()
        llm_provider = LLMProviderSpy(answer="test answer")
        orchestrator = CacheOrchestrator(
            embedder=embedder,
            llm_provider=llm_provider,
            vector_store=vector_store,
        )

        result = orchestrator.query("hello")

    assert result.response == "test answer"
    assert result.cache_hit is False
    assert llm_provider.calls_count == 1
    assert vector_store.insert_calls_count == 1
