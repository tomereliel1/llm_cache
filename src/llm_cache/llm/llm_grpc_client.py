from __future__ import annotations

from types import TracebackType
from typing import Self

import grpc

from llm_cache.llm import llm_pb2, llm_pb2_grpc
from llm_cache.llm.i_llm_provider import ILLMProvider


class LLMGrpcClient(ILLMProvider):
    """Client-side adapter that makes a remote LLM service look like ILLMProvider."""

    def __init__(
        self,
        target: str = "localhost:50053",
        timeout_seconds: float = 300.0,
        channel: grpc.Channel | None = None,
    ) -> None:
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = llm_pb2_grpc.LLMServiceStub(self._channel)

    def generate_answer(self, prompt: str) -> str:
        request = llm_pb2.GenerateRequest(prompt=prompt)

        try:
            reply = self._stub.Generate(request, timeout=self._timeout_seconds)
        except grpc.RpcError as error:
            code = error.code()
            code_name = code.name if code is not None else code
            raise RuntimeError(f"LLM gRPC call failed: {code_name}: {error.details()}") from error

        return reply.response

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
