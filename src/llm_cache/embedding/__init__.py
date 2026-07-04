from .grpc.client import EmbeddingGrpcClient
from .grpc.service import EmbeddingGrpcService
from .interface import IEmbedder

__all__ = [
    "EmbeddingGrpcClient",
    "EmbeddingGrpcService",
    "IEmbedder",
    "OllamaEmbedder",
]


def __getattr__(name: str):
    if name == "OllamaEmbedder":
        from .providers.ollama import OllamaEmbedder

        return OllamaEmbedder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
