from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc
import pytest

from llm_cache.llm import ILLMProvider, llm_pb2_grpc
from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.llm.llm_grpc_service import LLMGrpcService


class RecordingStub:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def Generate(self, request, timeout: float):
        self.timeout = timeout
        return type("Reply", (), {"response": "answer"})()


class RecordingLLMProvider(ILLMProvider):
    def __init__(self, answer: str = "remote answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate_answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class InvalidPromptLLMProvider(ILLMProvider):
    def generate_answer(self, prompt: str) -> str:
        raise ValueError("prompt must not be empty")


class FailingLLMProvider(ILLMProvider):
    def generate_answer(self, prompt: str) -> str:
        raise RuntimeError("provider unavailable")


@contextmanager
def running_llm_server(llm_provider: ILLMProvider) -> Iterator[str]:
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


def test_llm_grpc_client_returns_remote_response_and_forwards_prompt() -> None:
    provider = RecordingLLMProvider()

    with running_llm_server(provider) as target:
        with LLMGrpcClient(target=target) as client:
            response = client.generate_answer("hello")

    assert response == "remote answer"
    assert provider.prompts == ["hello"]


def test_llm_grpc_client_uses_five_minute_default_timeout() -> None:
    client = LLMGrpcClient()
    stub = RecordingStub()
    client._stub = stub

    try:
        assert client.generate_answer("hello") == "answer"
    finally:
        client.close()

    assert stub.timeout == 300.0


def test_llm_grpc_client_translates_invalid_argument_error() -> None:
    with running_llm_server(InvalidPromptLLMProvider()) as target:
        with LLMGrpcClient(target=target) as client:
            with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
                client.generate_answer("")


def test_llm_grpc_client_translates_internal_error() -> None:
    with running_llm_server(FailingLLMProvider()) as target:
        with LLMGrpcClient(target=target) as client:
            with pytest.raises(RuntimeError, match="INTERNAL.*provider unavailable"):
                client.generate_answer("hello")


def test_llm_grpc_client_translates_unavailable_server_error() -> None:
    with LLMGrpcClient(target="localhost:1", timeout_seconds=0.1) as client:
        with pytest.raises(RuntimeError, match="LLM gRPC call failed"):
            client.generate_answer("hello")
