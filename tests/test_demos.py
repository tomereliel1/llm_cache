from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc

from llm_cache.demos.grpc_providers_orchestrator_demo import (
    run_demo as run_grpc_providers_demo,
)
from llm_cache.demos.grpc_vector_store_orchestrator_demo import (
    local_vector_store_target,
    run_demo,
)
from llm_cache.demos.monolit_run import build_stub_demo_config
from llm_cache.embedding import embedding_pb2_grpc
from llm_cache.embedding.embedding_grpc_service import EmbeddingGrpcService
from llm_cache.llm import llm_pb2_grpc
from llm_cache.llm.llm_grpc_service import LLMGrpcService
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy
from llm_cache.vector_store import InMemoryVectorStore
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server


@contextmanager
def local_provider_targets() -> Iterator[tuple[str, str, str]]:
    embedding_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGrpcService(EmbedderStub()),
        embedding_server,
    )
    embedding_port = embedding_server.add_insecure_port("localhost:0")

    vector_store_server = create_vector_store_grpc_server(
        InMemoryVectorStore(similarity_threshold=0.8),
        max_workers=2,
    )
    vector_store_port = vector_store_server.add_insecure_port("localhost:0")

    llm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMGrpcService(LLMProviderSpy(answer="remote answer")),
        llm_server,
    )
    llm_port = llm_server.add_insecure_port("localhost:0")

    embedding_server.start()
    vector_store_server.start()
    llm_server.start()

    try:
        yield (
            f"localhost:{embedding_port}",
            f"localhost:{vector_store_port}",
            f"localhost:{llm_port}",
        )
    finally:
        embedding_server.stop(grace=0)
        vector_store_server.stop(grace=0)
        llm_server.stop(grace=0)


def test_monolit_run_demo_uses_stub_vector_store() -> None:
    config = build_stub_demo_config()

    assert config.vector_store.provider == "vector-store-miss-stub"
    assert config.llm.provider == "ollama"


def test_grpc_vector_store_orchestrator_demo_uses_cache() -> None:
    with local_vector_store_target() as target:
        first_hit, second_hit, llm_calls = run_demo("hello cache", target)

    assert first_hit is False
    assert second_hit is True
    assert llm_calls == 1


def test_grpc_providers_orchestrator_demo_uses_all_remote_providers() -> None:
    with local_provider_targets() as targets:
        embedding_target, vector_store_target, llm_target = targets
        first_hit, second_hit, response = run_grpc_providers_demo(
            prompt="hello remote cache",
            embedding_target=embedding_target,
            vector_store_target=vector_store_target,
            llm_target=llm_target,
        )

    assert first_hit is False
    assert second_hit is True
    assert response == "remote answer"
