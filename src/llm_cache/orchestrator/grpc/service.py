from __future__ import annotations

import logging

import grpc

from llm_cache.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_cache.orchestrator.grpc.generated import orchestrator_pb2, orchestrator_pb2_grpc
from llm_cache.orchestrator.orchestrator import CacheOrchestrator
from llm_cache.request_context import (
    new_request_id,
    request_context,
    request_id_from_grpc_context,
)

logger = logging.getLogger(__name__)


class OrchestratorGrpcService(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    """Adapt the public gRPC API to the cache orchestration algorithm."""

    def __init__(self, orchestrator: CacheOrchestrator) -> None:
        self._orchestrator = orchestrator

    def SubmitPrompt(
        self,
        request: orchestrator_pb2.SubmitPromptRequest,
        context: grpc.ServicerContext,
    ) -> orchestrator_pb2.SubmitPromptReply:
        if not request.prompt or not request.prompt.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Prompt must not be empty.")

        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
            logger.info("submit_prompt_received prompt_length=%s", len(request.prompt.strip()))
            try:
                result = self._orchestrator.query(request.prompt)
                logger.info("submit_prompt_completed cache_hit=%s", result.cache_hit)
            except ValueError as error:
                logger.warning("submit_prompt_invalid error=%s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except ProviderUnavailableError as error:
                logger.warning("submit_prompt_provider_unavailable error=%s", error)
                if error.technical_details:
                    context.set_trailing_metadata(
                        (("technical-details-bin", error.technical_details.encode("utf-8")),)
                    )
                context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
            except ProviderTimeoutError as error:
                logger.warning("submit_prompt_provider_timeout error=%s", error)
                if error.technical_details:
                    context.set_trailing_metadata(
                        (("technical-details-bin", error.technical_details.encode("utf-8")),)
                    )
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
            except RuntimeError as error:
                logger.exception("submit_prompt_runtime_error")
                # Provider gRPC clients use RuntimeError to preserve the failing
                # dependency, status code, and provider-side diagnostic. Forward
                # that context to the public client instead of hiding it behind a
                # generic orchestration failure.
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Failed to process query: {error}",
                )
            except Exception:
                logger.exception("submit_prompt_failed")
                context.abort(grpc.StatusCode.INTERNAL, "Failed to process query.")

        return orchestrator_pb2.SubmitPromptReply(
            response=result.response,
            cache_hit=result.cache_hit,
        )
