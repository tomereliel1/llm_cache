from .embedding_grpc_client import EmbeddingGrpcClient
from .embedding_grpc_service import EmbeddingGrpcService
from .i_embedder import IEmbedder

__all__ = [
    "EmbeddingGrpcClient",
    "EmbeddingGrpcService",
    "IEmbedder",
    "OllamaEmbedder",
]


def __getattr__(name: str):
    if name == "OllamaEmbedder":
        from .ollama_embedder import OllamaEmbedder

        return OllamaEmbedder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
