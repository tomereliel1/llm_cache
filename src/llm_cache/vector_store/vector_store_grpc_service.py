from concurrent import futures

import grpc

from llm_cache.vector_store.i_vector_store import IVectorStore
from llm_cache.vector_store.vector_store_pb2 import (
    SearchSimilarReply,
    SearchSimilarRequest,
    StoreReply,
    StoreRequest,
)
from llm_cache.vector_store.vector_store_pb2_grpc import (
    VectorStoreServiceServicer,
    add_VectorStoreServiceServicer_to_server,
)


class VectorStoreGrpcService(VectorStoreServiceServicer):
    """gRPC adapter that exposes an IVectorStore implementation."""

    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store

    def SearchSimilar(
        self,
        request: SearchSimilarRequest,
        context: grpc.ServicerContext,
    ) -> SearchSimilarReply:
        try:
            result = self._vector_store.search_similar(list(request.vector))
        except ValueError as error:
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
        try:
            self._vector_store.store(
                prompt=request.prompt,
                response=request.response,
                vector=list(request.vector),
            )
        except ValueError as error:
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
