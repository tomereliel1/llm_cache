from llm_cache.config.app_config import ConfigError, EmbeddingConfig
from llm_cache.embedding.i_embedder import IEmbedder
from llm_cache.embedding.ollama_embedder import OllamaEmbedder

_SUPPORTED_EMBEDDING_PROVIDERS = ("ollama",)


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def create_embedder(config: EmbeddingConfig) -> IEmbedder:
    provider = _normalize_provider(config.provider)

    if provider == "ollama":
        if not config.model:
            raise ConfigError(
                "Missing model for Ollama embedding provider. "
                "Example: EmbeddingConfig(provider='ollama', model='embeddinggemma')"
            )

        return OllamaEmbedder(model_name=config.model)

    raise ConfigError(
        f"Unknown embedding provider '{config.provider}'. "
        f"Supported embedding providers: {', '.join(_SUPPORTED_EMBEDDING_PROVIDERS)}"
    )
