import logging
from concurrent import futures

import grpc

from llm_cache.request_context import (
    new_request_id,
    request_context,
    request_id_from_grpc_context,
)
from llm_cache.vector_store.grpc.generated.vector_store_pb2 import (
    SearchSimilarReply,
    SearchSimilarRequest,
    StoreReply,
    StoreRequest,
)
from llm_cache.vector_store.grpc.generated.vector_store_pb2_grpc import (
    VectorStoreServiceServicer,
    add_VectorStoreServiceServicer_to_server,
)
from llm_cache.vector_store.interface import IVectorStore

logger = logging.getLogger(__name__)


class VectorStoreGrpcService(VectorStoreServiceServicer):
    """gRPC adapter that exposes an IVectorStore implementation."""

    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store

    def SearchSimilar(
        self,
        request: SearchSimilarRequest,
        context: grpc.ServicerContext,
    ) -> SearchSimilarReply:
        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
            logger.info("vector_search_received vector_size=%s", len(request.vector))
            try:
                result = self._vector_store.search_similar(list(request.vector))
                logger.info("vector_search_completed found=%s", result.found)
            except ValueError as error:
                logger.warning("vector_search_invalid error=%s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        return SearchSimilarReply(
            found=result.found,
            prompt=result.prompt,
            response=result.response,
        )

    def Store(
        self,
        request: StoreRequest,
        context: grpc.ServicerContext,
    ) -> StoreReply:
        request_id = request_id_from_grpc_context(context) or new_request_id()
        with request_context(request_id):
            logger.info(
                "vector_store_received prompt_length=%s response_length=%s vector_size=%s",
                len(request.prompt),
                len(request.response),
                len(request.vector),
            )
            try:
                self._vector_store.store(
                    prompt=request.prompt,
                    response=request.response,
                    vector=list(request.vector),
                )
                logger.info("vector_store_completed")
            except ValueError as error:
                logger.warning("vector_store_invalid error=%s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        return StoreReply(success=True)


def add_vector_store_to_server(vector_store: IVectorStore, server: grpc.Server) -> None:
    add_VectorStoreServiceServicer_to_server(
        VectorStoreGrpcService(vector_store=vector_store),
        server,
    )


def create_vector_store_grpc_server(
    vector_store: IVectorStore,
    max_workers: int = 10,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_vector_store_to_server(vector_store=vector_store, server=server)
    return server
