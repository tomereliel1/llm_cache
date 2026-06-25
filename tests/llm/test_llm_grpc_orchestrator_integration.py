from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc

from llm_cache.embedding import embedding_pb2_grpc
from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.embedding.embedding_grpc_service import EmbeddingGrpcService
from llm_cache.llm import llm_pb2_grpc
from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.llm.llm_grpc_service import LLMGrpcService
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy, VectorStoreMissStub


@contextmanager
def running_llm_server(llm_provider: LLMProviderSpy) -> Iterator[str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMGrpcService(llm_provider),
        server,
    )
    port = server.add_insecure_port("localhost:0")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        server.stop(grace=0)


@contextmanager
def running_embedding_server(embedder: EmbedderStub) -> Iterator[str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
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


def test_orchestrator_works_with_only_llm_over_grpc() -> None:
    provider = LLMProviderSpy(answer="remote answer")
    vector_store = VectorStoreMissStub()

    with running_llm_server(provider) as llm_target:
        with LLMGrpcClient(target=llm_target) as llm_client:
            orchestrator = CacheOrchestrator(
                embedder=EmbedderStub(),
                llm_provider=llm_client,
                vector_store=vector_store,
            )
            result = orchestrator.query("hello")

    assert result.response == "remote answer"
    assert result.cache_hit is False
    assert provider.calls_count == 1
    assert vector_store.insert_calls_count == 1


def test_orchestrator_works_with_embedding_and_llm_over_grpc() -> None:
    provider = LLMProviderSpy(answer="fully remote answer")
    vector_store = VectorStoreMissStub()

    with (
        running_embedding_server(EmbedderStub()) as embedding_target,
        running_llm_server(provider) as llm_target,
        EmbeddingGrpcClient(target=embedding_target) as embedder_client,
        LLMGrpcClient(target=llm_target) as llm_client,
    ):
        orchestrator = CacheOrchestrator(
            embedder=embedder_client,
            llm_provider=llm_client,
            vector_store=vector_store,
        )
        result = orchestrator.query("hello")

    assert result.response == "fully remote answer"
    assert result.cache_hit is False
    assert provider.calls_count == 1
    assert vector_store.insert_calls_count == 1
