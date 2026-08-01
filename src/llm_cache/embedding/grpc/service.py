from __future__ import annotations

import logging

import grpc

from llm_cache.embedding.grpc.generated import embedding_pb2, embedding_pb2_grpc
from llm_cache.embedding.interface import IEmbedder
from llm_cache.request_context import (
    new_request_id,
    request_context,
    request_id_from_grpc_context,
)

logger = logging.getLogger(__name__)


class EmbeddingGrpcService(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC service adapter for an ``IEmbedder`` implementation."""

    def __init__(self, embedder: IEmbedder) -> None:
        """Create the service around an embedder implementation.

        Args:
            embedder: Provider used to create prompt embeddings.
        """
        self._embedder = embedder

    def Embed(
        self,
        request: embedding_pb2.EmbedRequest,
        context: grpc.ServicerContext,
    ) -> embedding_pb2.EmbedReply:
        """Handle an embedding RPC request.

        Args:
            request: Protobuf request containing the prompt to embed.
            context: gRPC server context used for metadata and status handling.

        Returns:
            Protobuf reply containing the embedding vector.

        Raises:
            grpc.RpcError: Via ``context.abort`` for invalid input or provider failures.
        """
        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
            logger.info("embedding_request_received prompt_length=%s", len(request.prompt))
            try:
                vector = self._embedder.embed(request.prompt)
                logger.info("embedding_request_completed vector_size=%s", len(vector))
                return embedding_pb2.EmbedReply(vector=[float(value) for value in vector])
            except ValueError as error:
                logger.warning("embedding_request_invalid error=%s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except Exception as error:
                logger.exception("embedding_request_failed")
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Embedding generation failed: {error}",
                )
