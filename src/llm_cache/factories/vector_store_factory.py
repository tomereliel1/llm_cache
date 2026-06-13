from llm_cache.config.app_config import ConfigError, VectorStoreConfig
from llm_cache.config.provider_options import (
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.test_doubles.vector_store_hit_stub import VectorStoreHitStub
from llm_cache.test_doubles.vector_store_miss_stub import VectorStoreMissStub
from llm_cache.vector_store.in_memory_vector_store import InMemoryVectorStore
from llm_cache.vector_store.i_vector_store import IVectorStore


def create_vector_store(config: VectorStoreConfig) -> IVectorStore:
    provider = normalize_provider_name(config.provider)

    if provider == "vector-store-miss-stub":
        return VectorStoreMissStub(similarity_threshold=config.similarity_threshold)

    if provider == "vector-store-hit-stub":
        return VectorStoreHitStub(similarity_threshold=config.similarity_threshold)

    if provider == "in-memory":
        return InMemoryVectorStore(similarity_threshold=config.similarity_threshold)

    raise ConfigError(
        f"Unknown vector store provider '{config.provider}'. "
        f"Supported vector store providers: {', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)}"
    )
