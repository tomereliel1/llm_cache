from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    similarity_threshold: float = 0.8
    persist_path: str = ".cache/vector_store"
    collection_name: str = "llm_cache"
    max_capacity: int = 1000

    def __post_init__(self) -> None:
        if not 0 <= self.similarity_threshold <= 1:
            raise ConfigError(
                "similarity_threshold must be between 0 and 1. "
                f"Got: {self.similarity_threshold}"
            )

        if not self.persist_path.strip():
            raise ConfigError("persist_path must be a non-empty string")

        if not self.collection_name.strip():
            raise ConfigError("collection_name must be a non-empty string")

        if self.max_capacity < 1:
            raise ConfigError(f"max_capacity must be at least 1. Got: {self.max_capacity}")


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
