from .cache_entry_metadata import CacheEntryMetadata
from .i_eviction_policy import IEvictionPolicy
from .i_vector_store import IVectorStore, VectorStoreResult
from .in_memory_vector_store import InMemoryVectorStore
from .lru_eviction_policy import LRUEvictionPolicy
from .vector_store_grpc_client import VectorStoreGrpcClient
from .vector_store_grpc_service import VectorStoreGrpcService

__all__ = [
    "CacheEntryMetadata",
    "ChromaVectorStore",
    "IEvictionPolicy",
    "InMemoryVectorStore",
    "IVectorStore",
    "LRUEvictionPolicy",
    "VectorStoreGrpcClient",
    "VectorStoreGrpcService",
    "VectorStoreResult",
]


def __getattr__(name: str):
    if name == "ChromaVectorStore":
        from .chroma_vector_store import ChromaVectorStore

        return ChromaVectorStore

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
