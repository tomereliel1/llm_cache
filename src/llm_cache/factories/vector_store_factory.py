from llm_cache.config.app_config import ConfigError, VectorStoreConfig
from llm_cache.config.provider_options import (
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.test_doubles.vector_store_hit_stub import VectorStoreHitStub
from llm_cache.test_doubles.vector_store_miss_stub import VectorStoreMissStub
from llm_cache.vector_store.implementations.memory import InMemoryVectorStore
from llm_cache.vector_store.interface import IVectorStore

from .eviction_policy_factory import create_eviction_policy


def create_vector_store(config: VectorStoreConfig) -> IVectorStore:
    provider = normalize_provider_name(config.provider)
    if provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
        raise ConfigError(
            f"Unknown vector store provider '{config.provider}'. "
            f"Supported vector store providers: {', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)}"
        )

    eviction_policy = create_eviction_policy(
        config.eviction_policy,
        vector_store_provider=provider,
    )

    if provider == "vector-store-miss-stub":
        return VectorStoreMissStub(
            similarity_threshold=config.similarity_threshold,
            max_capacity=config.max_capacity,
        )

    if provider == "vector-store-hit-stub":
        return VectorStoreHitStub(
            similarity_threshold=config.similarity_threshold,
            max_capacity=config.max_capacity,
        )

    if provider == "in-memory":
        return InMemoryVectorStore(
            similarity_threshold=config.similarity_threshold,
            max_capacity=config.max_capacity,
            eviction_policy=eviction_policy,
        )

    if provider == "chroma":
        from llm_cache.vector_store.implementations.chroma import ChromaVectorStore

        distance_function = normalize_provider_name(config.distance_function)
        return ChromaVectorStore(
            similarity_threshold=config.similarity_threshold,
            persist_path=config.persist_path,
            collection_name=config.collection_name,
            max_capacity=config.max_capacity,
            eviction_policy=eviction_policy,
            distance_function=distance_function,
        )

    raise AssertionError(f"Unhandled vector store provider: {provider}")
