from .i_llm_provider import ILLMProvider
from .llm_grpc_client import LLMGrpcClient
from .llm_grpc_service import LLMGrpcService

__all__ = [
    "GroqLLMProvider",
    "ILLMProvider",
    "LLMGrpcClient",
    "LLMGrpcService",
    "OllamaLLMProvider",
]


def __getattr__(name: str):
    if name == "GroqLLMProvider":
        from .groq_llm_provider import GroqLLMProvider

        return GroqLLMProvider
    if name == "OllamaLLMProvider":
        from .ollama_llm_provider import OllamaLLMProvider

        return OllamaLLMProvider

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
