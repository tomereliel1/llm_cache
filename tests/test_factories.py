import pytest

from llm_cache.config import (
    AppConfig,
    ConfigError,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from llm_cache.embedding import OllamaEmbedder
from llm_cache.factories import (
    create_embedder,
    create_eviction_policy,
    create_llm_provider,
    create_vector_store,
)
from llm_cache.llm import GroqLLMProvider, OllamaLLMProvider
from llm_cache.test_doubles import VectorStoreHitStub, VectorStoreMissStub
from llm_cache.vector_store import InMemoryVectorStore, LRUEvictionPolicy


def _app_config_with_threshold(similarity_threshold: float) -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(provider="ollama", model="embeddinggemma"),
        llm=LLMConfig(provider="ollama", model="gemma3:4b"),
        vector_store=VectorStoreConfig(
            provider="vector-store-miss-stub",
            similarity_threshold=similarity_threshold,
        ),
    )


@pytest.mark.parametrize("similarity_threshold", [0, 0.8, 1])
def test_app_config_accepts_similarity_threshold_in_range(
    similarity_threshold: float,
) -> None:
    config = _app_config_with_threshold(similarity_threshold)

    assert config.vector_store.similarity_threshold == similarity_threshold


@pytest.mark.parametrize("similarity_threshold", [-0.1, 1.1])
def test_app_config_rejects_similarity_threshold_outside_range(
    similarity_threshold: float,
) -> None:
    with pytest.raises(ConfigError, match="similarity_threshold"):
        _app_config_with_threshold(similarity_threshold)


def test_create_embedder_returns_ollama_embedder_for_ollama() -> None:
    embedder = create_embedder(EmbeddingConfig(provider="ollama", model="embeddinggemma"))

    assert isinstance(embedder, OllamaEmbedder)


def test_create_embedder_normalizes_provider_name() -> None:
    embedder = create_embedder(EmbeddingConfig(provider=" Ollama ", model="embeddinggemma"))

    assert isinstance(embedder, OllamaEmbedder)


def test_create_embedder_rejects_missing_ollama_model() -> None:
    with pytest.raises(ConfigError, match="Missing model for Ollama embedding provider"):
        create_embedder(EmbeddingConfig(provider="ollama"))


def test_create_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigError, match="Unknown embedding provider 'unknown'.*ollama"):
        create_embedder(EmbeddingConfig(provider="unknown", model="embeddinggemma"))


def test_create_llm_provider_returns_ollama_llm_provider_for_ollama() -> None:
    llm_provider = create_llm_provider(LLMConfig(provider="ollama", model="gemma3:4b"))

    assert isinstance(llm_provider, OllamaLLMProvider)


def test_create_llm_provider_rejects_missing_ollama_model() -> None:
    with pytest.raises(ConfigError, match="Missing model for Ollama LLM provider"):
        create_llm_provider(LLMConfig(provider="ollama"))


def test_create_llm_provider_rejects_missing_groq_model() -> None:
    with pytest.raises(ConfigError, match="Missing model for Groq LLM provider"):
        create_llm_provider(LLMConfig(provider="groq"))


def test_create_llm_provider_rejects_missing_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="Missing Groq API key.*GROQ_API_KEY"):
        create_llm_provider(LLMConfig(provider="groq", model="llama-3.1-8b-instant"))


def test_create_llm_provider_returns_groq_llm_provider_for_groq_with_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_GROQ_API_KEY", "test-api-key")

    llm_provider = create_llm_provider(
        LLMConfig(
            provider="groq",
            model="llama-3.1-8b-instant",
            api_key_env="TEST_GROQ_API_KEY",
        )
    )

    assert isinstance(llm_provider, GroqLLMProvider)


def test_create_llm_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigError, match="Unknown LLM provider 'unknown'.*ollama, groq"):
        create_llm_provider(LLMConfig(provider="unknown", model="gemma3:4b"))


def test_create_vector_store_returns_vector_store_miss_stub() -> None:
    vector_store = create_vector_store(VectorStoreConfig(provider="vector-store-miss-stub"))

    assert isinstance(vector_store, VectorStoreMissStub)


def test_create_vector_store_returns_vector_store_hit_stub() -> None:
    vector_store = create_vector_store(VectorStoreConfig(provider="vector-store-hit-stub"))

    assert isinstance(vector_store, VectorStoreHitStub)
    assert vector_store.search_similar([0.1]).response == "cached answer"


def test_create_vector_store_returns_in_memory_vector_store() -> None:
    vector_store = create_vector_store(VectorStoreConfig(provider="in-memory"))

    assert isinstance(vector_store, InMemoryVectorStore)


def test_create_vector_store_returns_chroma_vector_store(tmp_path) -> None:
    from llm_cache.vector_store.chroma_vector_store import ChromaVectorStore

    vector_store = create_vector_store(
        VectorStoreConfig(provider="chroma", persist_path=str(tmp_path))
    )

    assert isinstance(vector_store, ChromaVectorStore)
    assert vector_store.eviction_policy is None


def test_create_vector_store_passes_similarity_threshold_to_provider() -> None:
    vector_store = create_vector_store(
        VectorStoreConfig(provider="in-memory", similarity_threshold=0.95)
    )

    assert isinstance(vector_store, InMemoryVectorStore)
    assert vector_store.similarity_threshold == 0.95


def test_create_vector_store_passes_chroma_config_to_provider(tmp_path) -> None:
    from llm_cache.vector_store.chroma_vector_store import ChromaVectorStore

    vector_store = create_vector_store(
        VectorStoreConfig(
            provider="chroma",
            similarity_threshold=0.95,
            persist_path=str(tmp_path),
            collection_name="factory_test",
            max_capacity=42,
            eviction_policy="lru",
        )
    )

    assert isinstance(vector_store, ChromaVectorStore)
    assert vector_store.similarity_threshold == 0.95
    assert vector_store.persist_path == str(tmp_path)
    assert vector_store.collection_name == "factory_test"
    assert vector_store.max_capacity == 42
    assert isinstance(vector_store.eviction_policy, LRUEvictionPolicy)


def test_create_eviction_policy_returns_lru_policy() -> None:
    policy = create_eviction_policy("lru")

    assert isinstance(policy, LRUEvictionPolicy)


def test_create_eviction_policy_returns_none_for_default_policy() -> None:
    policy = create_eviction_policy("default")

    assert policy is None


def test_create_eviction_policy_normalizes_policy_name() -> None:
    policy = create_eviction_policy(" LRU ")

    assert isinstance(policy, LRUEvictionPolicy)


def test_create_eviction_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ConfigError, match="Unknown eviction policy 'unknown'.*lru"):
        create_eviction_policy("unknown")


def test_create_vector_store_normalizes_provider_name() -> None:
    vector_store = create_vector_store(VectorStoreConfig(provider=" Vector-Store-Miss-Stub "))

    assert isinstance(vector_store, VectorStoreMissStub)


def test_create_vector_store_rejects_unknown_provider() -> None:
    with pytest.raises(
        ConfigError,
        match="Unknown vector store provider 'unknown'.*vector-store-miss-stub",
    ):
        create_vector_store(VectorStoreConfig(provider="unknown"))
