from .eviction.interface import IEvictionPolicy
from .eviction.lru import LRUEvictionPolicy
from .grpc.client import VectorStoreGrpcClient
from .grpc.service import VectorStoreGrpcService
from .implementations.memory import InMemoryVectorStore
from .interface import IVectorStore, VectorStoreResult
from .models import CacheEntryMetadata

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
        from .implementations.chroma import ChromaVectorStore

        return ChromaVectorStore

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
