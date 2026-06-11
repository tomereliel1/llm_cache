import pytest

from llm_cache.embedding import OllamaEmbedder
from llm_cache.llm import GroqLLMProvider, OllamaLLMProvider
from llm_cache.test_doubles import VectorStoreHitStub, VectorStoreMissStub


def test_vector_store_miss_stub_health_check_is_healthy() -> None:
    result = VectorStoreMissStub().health_check()

    assert result.healthy is True
    assert result.name == "vector-store:miss-stub"


def test_vector_store_hit_stub_health_check_is_healthy() -> None:
    result = VectorStoreHitStub().health_check()

    assert result.healthy is True
    assert result.name == "vector-store:hit-stub"


def test_groq_provider_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        GroqLLMProvider()


def test_groq_health_check_passes_when_api_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")

    result = GroqLLMProvider().health_check()

    assert result.healthy is True
    assert result.name == "llm:groq"


def test_ollama_embedder_health_check_passes_when_embedding_is_non_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = OllamaEmbedder("embeddinggemma")
    monkeypatch.setattr(embedder, "embed", lambda prompt: [0.1, 0.2])

    result = embedder.health_check()

    assert result.healthy is True
    assert result.name == "embedding:ollama"


def test_ollama_embedder_health_check_fails_when_embed_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = OllamaEmbedder("embeddinggemma")

    def raise_error(prompt: str) -> list[float]:
        raise RuntimeError("no ollama")

    monkeypatch.setattr(embedder, "embed", raise_error)

    result = embedder.health_check()

    assert result.healthy is False
    assert result.details == "no ollama"


def test_ollama_llm_health_check_passes_when_response_is_non_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaLLMProvider("gemma3:4b")
    monkeypatch.setattr(provider, "_generate_answer", lambda prompt: "OK")

    result = provider.health_check()

    assert result.healthy is True
    assert result.name == "llm:ollama"


def test_ollama_llm_health_check_fails_when_generate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaLLMProvider("gemma3:4b")

    def raise_error(prompt: str) -> str:
        raise RuntimeError("no ollama")

    monkeypatch.setattr(provider, "_generate_answer", raise_error)

    result = provider.health_check()

    assert result.healthy is False
    assert result.details == "no ollama"
