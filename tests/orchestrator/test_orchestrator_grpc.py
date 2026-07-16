from concurrent import futures
from contextlib import ExitStack

import grpc
import pytest

from llm_cache.embedding.grpc.client import EmbeddingGrpcClient
from llm_cache.embedding.grpc.generated import embedding_pb2_grpc
from llm_cache.embedding.grpc.service import EmbeddingGrpcService
from llm_cache.errors import ProviderUnavailableError
from llm_cache.llm.grpc.client import LLMGrpcClient
from llm_cache.llm.grpc.generated import llm_pb2_grpc
from llm_cache.llm.grpc.service import LLMGrpcService
from llm_cache.orchestrator import (
    CacheOrchestrator,
    OrchestratorGrpcClient,
)
from llm_cache.orchestrator.grpc.generated import orchestrator_pb2, orchestrator_pb2_grpc
from llm_cache.orchestrator.grpc.server import create_orchestrator_server
from llm_cache.orchestrator.grpc.service import OrchestratorGrpcService
from llm_cache.request_context import get_current_request_id
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy
from llm_cache.vector_store import InMemoryVectorStore
from llm_cache.vector_store.grpc.client import VectorStoreGrpcClient
from llm_cache.vector_store.grpc.service import create_vector_store_grpc_server


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


class RequestIdRecordingEmbedder:
    def __init__(self) -> None:
        self.request_ids = []

    def embed(self, prompt):
        self.request_ids.append(get_current_request_id())
        return [1.0, 0.0]


class RequestIdRecordingLLMProvider:
    def __init__(self) -> None:
        self.request_ids = []
        self.calls_count = 0

    def generate_answer(self, prompt):
        self.request_ids.append(get_current_request_id())
        self.calls_count += 1
        return "answer from remote LLM"


class RequestIdRecordingVectorStore:
    def __init__(self) -> None:
        self.request_ids_by_operation = []
        self._vector_store = InMemoryVectorStore(similarity_threshold=0.8)

    def search_similar(self, vector):
        self.request_ids_by_operation.append(("search", get_current_request_id()))
        return self._vector_store.search_similar(vector)

    def store(self, prompt, response, vector):
        self.request_ids_by_operation.append(("store", get_current_request_id()))
        return self._vector_store.store(prompt, response, vector)


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
    embedder = RequestIdRecordingEmbedder()
    vector_store = RequestIdRecordingVectorStore()
    llm = RequestIdRecordingLLMProvider()
    embedding_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGrpcService(embedder), embedding_server
    )
    embedding_port = embedding_server.add_insecure_port("localhost:0")

    vector_server = create_vector_store_grpc_server(
        vector_store, max_workers=2
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
    assert embedder.request_ids[0] == vector_store.request_ids_by_operation[0][1]
    assert embedder.request_ids[0] == llm.request_ids[0]
    assert embedder.request_ids[0] == vector_store.request_ids_by_operation[1][1]
    assert embedder.request_ids[1] == vector_store.request_ids_by_operation[2][1]
    assert embedder.request_ids[0] != embedder.request_ids[1]


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("bad prompt"), grpc.StatusCode.INVALID_ARGUMENT),
        (
            ProviderUnavailableError("LLM", "llm-service:50053", "DNS failure"),
            grpc.StatusCode.UNAVAILABLE,
        ),
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
    if expected_code is grpc.StatusCode.UNAVAILABLE:
        assert (
            exc_info.value.details()
            == "LLM service is unavailable. Check that it is running and reachable, "
            "then try again."
        )
    elif expected_code is grpc.StatusCode.INTERNAL:
        assert exc_info.value.details() == "Failed to process query: secret detail"


def test_public_client_preserves_clear_llm_unavailable_error() -> None:
    orchestrator = RecordingOrchestrator(
        ProviderUnavailableError("LLM", "llm-service:50053", "DNS failure")
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcService(orchestrator), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()

    try:
        with OrchestratorGrpcClient(f"localhost:{port}") as client:
            with pytest.raises(
                RuntimeError,
                match=r"^LLM service is unavailable\.",
            ) as exc_info:
                client.query("hello")
        assert "Provider: LLM" in exc_info.value.technical_details
        assert "Target: llm-service:50053" in exc_info.value.technical_details
        assert "Details: DNS failure" in exc_info.value.technical_details
    finally:
        server.stop(grace=0)


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
