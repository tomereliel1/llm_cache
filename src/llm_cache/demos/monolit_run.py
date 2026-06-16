from __future__ import annotations

from llm_cache.config import AppConfig, EmbeddingConfig, LLMConfig, VectorStoreConfig
from llm_cache.config.provider_options import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_PROMPT,
    default_embedding_model,
    default_llm_model,
)
from llm_cache.factories import create_embedder, create_llm_provider, create_vector_store
from llm_cache.orchestrator import CacheOrchestrator, QueryResult


def build_stub_demo_config() -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(
            provider=DEFAULT_EMBEDDING_PROVIDER,
            model=default_embedding_model(DEFAULT_EMBEDDING_PROVIDER),
        ),
        llm=LLMConfig(
            provider=DEFAULT_LLM_PROVIDER,
            model=default_llm_model(DEFAULT_LLM_PROVIDER),
        ),
        vector_store=VectorStoreConfig(provider="vector-store-miss-stub"),
    )


def build_orchestrator(config: AppConfig) -> CacheOrchestrator:
    return CacheOrchestrator(
        embedder=create_embedder(config.embedding),
        llm_provider=create_llm_provider(config.llm),
        vector_store=create_vector_store(config.vector_store),
    )


def print_result(prompt: str, result: QueryResult) -> None:
    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Response:")
    print(result.response)

    print()
    print("Metadata:")
    print(f"cache_hit: {result.cache_hit}")


def main() -> None:
    config = build_stub_demo_config()
    orchestrator = build_orchestrator(config)
    result = orchestrator.query(DEFAULT_PROMPT)
    print_result(DEFAULT_PROMPT, result)


if __name__ == "__main__":
    main()
