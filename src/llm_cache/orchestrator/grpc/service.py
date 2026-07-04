from __future__ import annotations

import grpc

from llm_cache.orchestrator.grpc.generated import orchestrator_pb2, orchestrator_pb2_grpc
from llm_cache.orchestrator.orchestrator import CacheOrchestrator


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

        try:
            result = self._orchestrator.query(request.prompt)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to process query.")

        return orchestrator_pb2.SubmitPromptReply(
            response=result.response,
            cache_hit=result.cache_hit,
        )
