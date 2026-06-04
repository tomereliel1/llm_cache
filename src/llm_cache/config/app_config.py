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


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    similarity_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not 0 <= self.similarity_threshold <= 1:
            raise ConfigError(
                "similarity_threshold must be between 0 and 1. "
                f"Got: {self.similarity_threshold}"
            )
