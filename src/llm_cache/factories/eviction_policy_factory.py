from llm_cache.config.app_config import ConfigError
from llm_cache.config.provider_options import (
    SUPPORTED_EVICTION_POLICIES,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.vector_store import IEvictionPolicy, LRUEvictionPolicy


def create_eviction_policy(
    policy: str,
    *,
    vector_store_provider: str,
) -> IEvictionPolicy | None:
    normalized_provider = normalize_provider_name(vector_store_provider)
    if normalized_provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
        raise ConfigError(
            f"Unknown vector store provider '{vector_store_provider}'. "
            "Supported vector store providers: "
            f"{', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)}"
        )

    normalized_policy = normalize_provider_name(policy)
    if normalized_policy not in SUPPORTED_EVICTION_POLICIES:
        raise ConfigError(
            f"Unknown eviction policy '{policy}'. "
            f"Supported eviction policies: {', '.join(SUPPORTED_EVICTION_POLICIES)}"
        )

    provider_option = SUPPORTED_VECTOR_STORE_PROVIDERS[normalized_provider]
    if normalized_policy not in provider_option.supported_eviction_policies:
        raise ConfigError(
            f"Eviction policy '{policy}' is not supported by vector store provider "
            f"'{vector_store_provider}'. Supported eviction policies: "
            f"{', '.join(provider_option.supported_eviction_policies)}"
        )

    if normalized_policy == "default":
        normalized_policy = provider_option.default_eviction_policy
        if normalized_policy == "default":
            return None

    if normalized_policy == "lru":
        return LRUEvictionPolicy()

    raise AssertionError(
        f"Unhandled eviction policy '{normalized_policy}' configured for "
        f"vector store provider '{normalized_provider}'"
    )
