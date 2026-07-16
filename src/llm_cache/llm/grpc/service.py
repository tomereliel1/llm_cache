from __future__ import annotations

import grpc

from llm_cache.llm.grpc.generated import llm_pb2, llm_pb2_grpc
from llm_cache.llm.interface import ILLMProvider
from llm_cache.request_context import (
    new_request_id,
    request_context,
    request_id_from_grpc_context,
)


class LLMGrpcService(llm_pb2_grpc.LLMServiceServicer):
    """Server-side adapter from protobuf requests to the internal LLM interface."""

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    def Generate(
        self,
        request: llm_pb2.GenerateRequest,
        context: grpc.ServicerContext,
    ) -> llm_pb2.GenerateReply:
        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
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
