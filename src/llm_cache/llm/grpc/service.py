from __future__ import annotations

import logging

import grpc

from llm_cache.llm.grpc.generated import llm_pb2, llm_pb2_grpc
from llm_cache.llm.interface import ILLMProvider
from llm_cache.request_context import (
    new_request_id,
    request_context,
    request_id_from_grpc_context,
)

logger = logging.getLogger(__name__)


class LLMGrpcService(llm_pb2_grpc.LLMServiceServicer):
    """gRPC service adapter for an ``ILLMProvider`` implementation."""

    def __init__(self, llm_provider: ILLMProvider) -> None:
        """Create the service around an LLM provider implementation.

        Args:
            llm_provider: Provider used to generate answers.
        """
        self._llm_provider = llm_provider

    def Generate(
        self,
        request: llm_pb2.GenerateRequest,
        context: grpc.ServicerContext,
    ) -> llm_pb2.GenerateReply:
        """Handle an LLM generation RPC request.

        Args:
            request: Protobuf request containing the prompt.
            context: gRPC server context used for metadata and status handling.

        Returns:
            Protobuf reply containing the generated response.

        Raises:
            grpc.RpcError: Via ``context.abort`` for invalid input or provider failures.
        """
        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
            logger.info("llm_request_received prompt_length=%s", len(request.prompt))
            try:
                response = self._llm_provider.generate_answer(request.prompt)
                logger.info("llm_request_completed response_length=%s", len(response))
                return llm_pb2.GenerateReply(response=response)
            except ValueError as error:
                logger.warning("llm_request_invalid error=%s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except Exception as error:
                logger.exception("llm_request_failed")
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"LLM generation failed: {error}",
                )
