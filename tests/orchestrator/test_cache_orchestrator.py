from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import (
    EmbedderStub,
    LLMProviderSpy,
    VectorStoreHitStub,
    VectorStoreMissStub,
)


def test_query_on_cache_miss_generates_and_stores_response() -> None:
    llm_provider = LLMProviderSpy()
    vector_store = VectorStoreMissStub()
    orchestrator = CacheOrchestrator(
        embedder=EmbedderStub(),
        llm_provider=llm_provider,
        vector_store=vector_store,
    )

    result = orchestrator.query("What is the capital of Israel?")

    assert result.cache_hit is False
    assert result.response == "generated answer"
    assert llm_provider.calls_count == 1
    assert vector_store.insert_calls_count == 1


def test_query_on_cache_hit_returns_cached_response_without_llm_or_insert() -> None:
    llm_provider = LLMProviderSpy()
    vector_store = VectorStoreHitStub(cached_response="cached answer")
    orchestrator = CacheOrchestrator(
        embedder=EmbedderStub(),
        llm_provider=llm_provider,
        vector_store=vector_store,
    )

    result = orchestrator.query("What is the capital of Israel?")

    assert result.cache_hit is True
    assert result.response == "cached answer"
    assert llm_provider.calls_count == 0
    assert vector_store.insert_calls_count == 0
