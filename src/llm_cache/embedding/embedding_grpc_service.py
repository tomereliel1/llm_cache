from __future__ import annotations

import grpc

from llm_cache.embedding import embedding_pb2, embedding_pb2_grpc
from llm_cache.embedding.i_embedder import IEmbedder


class EmbeddingGrpcService(embedding_pb2_grpc.EmbeddingServiceServicer):
    """Server-side adapter from protobuf requests to the internal embedder interface."""

    def __init__(self, embedder: IEmbedder) -> None:
        self._embedder = embedder

    def Embed(
        self,
        request: embedding_pb2.EmbedRequest,
        context: grpc.ServicerContext,
    ) -> embedding_pb2.EmbedReply:
        try:
            vector = self._embedder.embed(request.prompt)
            return embedding_pb2.EmbedReply(vector=[float(value) for value in vector])
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Embedding generation failed: {error}",
            )
