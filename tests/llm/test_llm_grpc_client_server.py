from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc
import pytest

from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.llm import ILLMProvider
from llm_cache.llm.grpc.client import LLMGrpcClient
from llm_cache.llm.grpc.generated import llm_pb2_grpc
from llm_cache.llm.grpc.service import LLMGrpcService


class RecordingStub:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def Generate(self, request, timeout: float):
        self.timeout = timeout
        return type("Reply", (), {"response": "answer"})()


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
    def Generate(self, request, timeout: float):
        raise DeadlineExceededError()


class UnavailableStub:
    def Generate(self, request, timeout: float):
        raise UnavailableError()


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
    client = LLMGrpcClient(target="localhost:1", timeout_seconds=0.1)
    client._stub = UnavailableStub()

    try:
        with pytest.raises(
            ProviderUnavailableError,
            match=r"^LLM service is unavailable\.",
        ) as exc_info:
            client.generate_answer("hello")
    finally:
        client.close()

    assert exc_info.value.technical_details is not None
    assert "gRPC status: UNAVAILABLE" in exc_info.value.technical_details


def test_llm_grpc_client_translates_deadline_exceeded_error() -> None:
    client = LLMGrpcClient(target="localhost:50053", timeout_seconds=0.5)
    client._stub = DeadlineExceededStub()

    try:
        with pytest.raises(
            ProviderTimeoutError,
            match=r"Increase the provider timeout using --provider-timeout-seconds",
        ) as exc_info:
            client.generate_answer("hello")
    finally:
        client.close()

    assert "gRPC status: DEADLINE_EXCEEDED" in exc_info.value.technical_details
    assert "Timeout seconds: 0.5" in exc_info.value.technical_details
