from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProviderOption:
    name: str
    default_model: str
    supported_models: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ProviderOption:
    name: str
    description: str


DEFAULT_PROMPT = "What is the capital of Israel?"
DEFAULT_SIMILARITY_THRESHOLD = 0.8

DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_VECTOR_STORE_PROVIDER = "vector-store-miss-stub"


SUPPORTED_EMBEDDING_PROVIDERS: dict[str, ModelProviderOption] = {
    "ollama": ModelProviderOption(
        name="ollama",
        default_model="embeddinggemma",
        supported_models=("embeddinggemma",),
        description="Local Ollama embedding provider.",
    ),
}


SUPPORTED_LLM_PROVIDERS: dict[str, ModelProviderOption] = {
    "ollama": ModelProviderOption(
        name="ollama",
        default_model="gemma3:4b",
        supported_models=("gemma3:4b",),
        description="Local Ollama LLM provider.",
    ),
    "groq": ModelProviderOption(
        name="groq",
        default_model="llama-3.1-8b-instant",
        supported_models=("llama-3.1-8b-instant",),
        description="Groq-hosted LLM provider.",
    ),
}


SUPPORTED_VECTOR_STORE_PROVIDERS: dict[str, ProviderOption] = {
    "vector-store-miss-stub": ProviderOption(
        name="vector-store-miss-stub",
        description="Test double vector store that always returns a cache miss.",
    ),
    "vector-store-hit-stub": ProviderOption(
        name="vector-store-hit-stub",
        description="Test double vector store that always returns a cache hit.",
    ),
    "in-memory": ProviderOption(
        name="in-memory",
        description="Local in-memory vector store using cosine similarity.",
    ),
}


def normalize_provider_name(provider: str) -> str:
    return provider.strip().lower()


def default_embedding_model(provider: str) -> str:
    normalized = normalize_provider_name(provider)
    return SUPPORTED_EMBEDDING_PROVIDERS[normalized].default_model


def default_llm_model(provider: str) -> str:
    normalized = normalize_provider_name(provider)
    return SUPPORTED_LLM_PROVIDERS[normalized].default_model


def format_supported_configs() -> str:
    lines: list[str] = [
        "Supported configurations",
        "========================",
        "",
        "Embedding providers:",
    ]

    for provider in SUPPORTED_EMBEDDING_PROVIDERS.values():
        lines.extend(
            [
                f"  - {provider.name}",
                f"    description: {provider.description}",
                f"    default model: {provider.default_model}",
                "    supported models:",
            ]
        )
        lines.extend(f"      - {model}" for model in provider.supported_models)
        lines.append("")

    lines.append("LLM providers:")
    for provider in SUPPORTED_LLM_PROVIDERS.values():
        lines.extend(
            [
                f"  - {provider.name}",
                f"    description: {provider.description}",
                f"    default model: {provider.default_model}",
                "    supported models:",
            ]
        )
        lines.extend(f"      - {model}" for model in provider.supported_models)
        lines.append("")

    lines.append("Vector-store providers:")
    for provider in SUPPORTED_VECTOR_STORE_PROVIDERS.values():
        lines.extend(
            [
                f"  - {provider.name}",
                f"    description: {provider.description}",
                "",
            ]
        )

    lines.extend(
        [
            "Global defaults:",
            f"  prompt: {DEFAULT_PROMPT!r}",
            f"  embedding provider: {DEFAULT_EMBEDDING_PROVIDER}",
            f"  embedding model: {default_embedding_model(DEFAULT_EMBEDDING_PROVIDER)}",
            f"  llm provider: {DEFAULT_LLM_PROVIDER}",
            f"  llm model: {default_llm_model(DEFAULT_LLM_PROVIDER)}",
            f"  vector-store provider: {DEFAULT_VECTOR_STORE_PROVIDER}",
            f"  similarity threshold: {DEFAULT_SIMILARITY_THRESHOLD}",
        ]
    )

    return "\n".join(lines)
