from llm_cache.config.app_config import ConfigError, EmbeddingConfig
from llm_cache.config.provider_options import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.embedding.i_embedder import IEmbedder
from llm_cache.embedding.ollama_embedder import OllamaEmbedder


def create_embedder(config: EmbeddingConfig) -> IEmbedder:
    provider = normalize_provider_name(config.provider)

    if provider == "ollama":
        if not config.model:
            raise ConfigError(
                "Missing model for Ollama embedding provider. "
                "Example: EmbeddingConfig(provider='ollama', model='embeddinggemma')"
            )

        return OllamaEmbedder(model_name=config.model)

    raise ConfigError(
        f"Unknown embedding provider '{config.provider}'. "
        f"Supported embedding providers: {', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}"
    )
