from llm_cache.config.app_config import ConfigError
from llm_cache.config.provider_options import (
    SUPPORTED_EVICTION_POLICIES,
    default_vector_store_eviction_policy,
    normalize_provider_name,
)
from llm_cache.vector_store import IEvictionPolicy, LRUEvictionPolicy


def create_eviction_policy(
    policy: str,
    vector_store_provider: str | None = None,
) -> IEvictionPolicy | None:
    normalized = normalize_provider_name(policy)

    if normalized == "default":
        if vector_store_provider is None:
            return None

        normalized = default_vector_store_eviction_policy(vector_store_provider)
        if normalized == "default":
            return None

    if normalized == "lru":
        return LRUEvictionPolicy()

    raise ConfigError(
        f"Unknown eviction policy '{policy}'. "
        f"Supported eviction policies: {', '.join(SUPPORTED_EVICTION_POLICIES)}"
    )
