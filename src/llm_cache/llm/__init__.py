from .grpc.client import LLMGrpcClient
from .grpc.service import LLMGrpcService
from .interface import ILLMProvider

__all__ = [
    "GroqLLMProvider",
    "ILLMProvider",
    "LLMGrpcClient",
    "LLMGrpcService",
    "OllamaLLMProvider",
]


def __getattr__(name: str):
    if name == "GroqLLMProvider":
        from .providers.groq import GroqLLMProvider

        return GroqLLMProvider
    if name == "OllamaLLMProvider":
        from .providers.ollama import OllamaLLMProvider

        return OllamaLLMProvider

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
