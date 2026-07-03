from concurrent import futures
from contextlib import ExitStack

import grpc
import pytest

from llm_cache.embedding import embedding_pb2_grpc
from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.embedding.embedding_grpc_service import EmbeddingGrpcService
from llm_cache.llm import llm_pb2_grpc
from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.llm.llm_grpc_service import LLMGrpcService
from llm_cache.orchestrator import (
    CacheOrchestrator,
    OrchestratorGrpcClient,
    orchestrator_pb2,
    orchestrator_pb2_grpc,
)
from llm_cache.orchestrator.orchestrator_grpc_service import OrchestratorGrpcService
from llm_cache.orchestrator.orchestrator_server import create_orchestrator_server
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy
from llm_cache.vector_store import InMemoryVectorStore
from llm_cache.vector_store.vector_store_grpc_client import VectorStoreGrpcClient
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server


class RecordingOrchestrator:
    def __init__(self, error=None):
        self.prompts = []
        self.error = error

    def query(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        from llm_cache.orchestrator import QueryResult

        return QueryResult("answer", True)


@pytest.fixture
def grpc_target():
    llm = LLMProviderSpy(answer="answer")
    orchestrator = CacheOrchestrator(
        EmbedderStub(vector=[1.0, 0.0]),
        llm,
        InMemoryVectorStore(similarity_threshold=0.8),
    )
    server = create_orchestrator_server(orchestrator, max_workers=2)
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        yield f"localhost:{port}", llm
    finally:
        server.stop(grace=0)


def test_public_grpc_service_observes_miss_then_hit(grpc_target):
    target, llm = grpc_target
    with OrchestratorGrpcClient(target) as client:
        first = client.query("same prompt")
        second = client.query("same prompt")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response == "answer"
    assert llm.calls_count == 1


def test_fully_distributed_grpc_chain_observes_miss_then_hit():
    llm = LLMProviderSpy(answer="answer from remote LLM")
    embedding_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGrpcService(EmbedderStub(vector=[1.0, 0.0])), embedding_server
    )
    embedding_port = embedding_server.add_insecure_port("localhost:0")

    vector_server = create_vector_store_grpc_server(
        InMemoryVectorStore(similarity_threshold=0.8), max_workers=2
    )
    vector_port = vector_server.add_insecure_port("localhost:0")

    llm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(LLMGrpcService(llm), llm_server)
    llm_port = llm_server.add_insecure_port("localhost:0")
    provider_servers = [embedding_server, vector_server, llm_server]
    for server in provider_servers:
        server.start()

    with ExitStack() as stack:
        orchestrator = CacheOrchestrator(
            stack.enter_context(EmbeddingGrpcClient(f"localhost:{embedding_port}")),
            stack.enter_context(LLMGrpcClient(f"localhost:{llm_port}")),
            stack.enter_context(VectorStoreGrpcClient(f"localhost:{vector_port}")),
        )
        public_server = create_orchestrator_server(orchestrator, max_workers=2)
        public_port = public_server.add_insecure_port("localhost:0")
        public_server.start()
        try:
            with OrchestratorGrpcClient(f"localhost:{public_port}") as client:
                first = client.query("same prompt")
                second = client.query("same prompt")
        finally:
            public_server.stop(grace=0)
            for server in provider_servers:
                server.stop(grace=0)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response == "answer from remote LLM"
    assert llm.calls_count == 1


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("bad prompt"), grpc.StatusCode.INVALID_ARGUMENT),
        (RuntimeError("secret detail"), grpc.StatusCode.INTERNAL),
    ],
)
def test_service_maps_orchestrator_errors(error, expected_code):
    orchestrator = RecordingOrchestrator(error)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcService(orchestrator), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.SubmitPrompt(orchestrator_pb2.SubmitPromptRequest(prompt="hello"))
    channel.close()
    server.stop(grace=0)

    assert exc_info.value.code() == expected_code
    if expected_code is grpc.StatusCode.INTERNAL:
        assert exc_info.value.details() == "Failed to process query."


def test_empty_prompt_is_rejected_before_orchestrator_call():
    orchestrator = RecordingOrchestrator()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcService(orchestrator), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.SubmitPrompt(orchestrator_pb2.SubmitPromptRequest(prompt="  "))
    channel.close()
    server.stop(grace=0)

    assert exc_info.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert orchestrator.prompts == []
