import grpc
import pytest

from llm_cache.vector_store.in_memory_vector_store import InMemoryVectorStore
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server
from llm_cache.vector_store.vector_store_pb2 import SearchSimilarRequest, StoreRequest
from llm_cache.vector_store.vector_store_pb2_grpc import VectorStoreServiceStub


@pytest.fixture
def vector_store_stub() -> VectorStoreServiceStub:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    server = create_vector_store_grpc_server(vector_store)
    port = server.add_insecure_port("localhost:0")
    server.start()

    with grpc.insecure_channel(f"localhost:{port}") as channel:
        yield VectorStoreServiceStub(channel)

    server.stop(grace=None)


def test_store_and_search_similar_over_grpc(vector_store_stub: VectorStoreServiceStub) -> None:
    store_reply = vector_store_stub.Store(
        StoreRequest(
            prompt="What is semantic caching?",
            response="cached response",
            vector=[1.0, 0.0],
        )
    )

    search_reply = vector_store_stub.SearchSimilar(SearchSimilarRequest(vector=[0.9, 0.1]))

    assert store_reply.success is True
    assert search_reply.found is True
    assert search_reply.prompt == "What is semantic caching?"
    assert search_reply.response == "cached response"


def test_search_similar_returns_miss_over_grpc(
    vector_store_stub: VectorStoreServiceStub,
) -> None:
    search_reply = vector_store_stub.SearchSimilar(SearchSimilarRequest(vector=[1.0, 0.0]))

    assert search_reply.found is False
    assert search_reply.prompt == ""
    assert search_reply.response == ""


def test_store_validation_errors_return_invalid_argument(
    vector_store_stub: VectorStoreServiceStub,
) -> None:
    with pytest.raises(grpc.RpcError) as error:
        vector_store_stub.Store(
            StoreRequest(
                prompt="prompt",
                response="response",
                vector=[0.0, 0.0],
            )
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "zero vectors" in error.value.details()
