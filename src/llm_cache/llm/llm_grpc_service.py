from __future__ import annotations

import grpc

from llm_cache.llm import llm_pb2, llm_pb2_grpc
from llm_cache.llm.i_llm_provider import ILLMProvider


class LLMGrpcService(llm_pb2_grpc.LLMServiceServicer):
    """Server-side adapter from protobuf requests to the internal LLM interface."""

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    def Generate(
        self,
        request: llm_pb2.GenerateRequest,
        context: grpc.ServicerContext,
    ) -> llm_pb2.GenerateReply:
        try:
            response = self._llm_provider.generate_answer(request.prompt)
            return llm_pb2.GenerateReply(response=response)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"LLM generation failed: {error}",
            )
