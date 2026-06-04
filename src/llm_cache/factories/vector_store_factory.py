from llm_cache.config.app_config import ConfigError, VectorStoreConfig
from llm_cache.test_doubles.vector_store_hit_stub import VectorStoreHitStub
from llm_cache.test_doubles.vector_store_miss_stub import VectorStoreMissStub
from llm_cache.vector_store.i_vector_store import IVectorStore

_SUPPORTED_VECTOR_STORE_PROVIDERS = ("vector-store-miss-stub", "vector-store-hit-stub")


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def create_vector_store(config: VectorStoreConfig) -> IVectorStore:
    provider = _normalize_provider(config.provider)

    if provider == "vector-store-miss-stub":
        return VectorStoreMissStub()

    if provider == "vector-store-hit-stub":
        return VectorStoreHitStub()

    raise ConfigError(
        f"Unknown vector store provider '{config.provider}'. "
        f"Supported vector store providers: {', '.join(_SUPPORTED_VECTOR_STORE_PROVIDERS)}"
    )
