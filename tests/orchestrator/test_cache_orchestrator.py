from llm_cache.embedding import IEmbedder
from llm_cache.llm import ILLMProvider
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import VectorStoreHitStub, VectorStoreMissStub


class LocalFakeEmbedder(IEmbedder):
    def embed(self, prompt: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class LocalSpyLLMProvider(ILLMProvider):
    def __init__(self, answer: str = "generated answer") -> None:
        self.answer = answer
        self.calls_count = 0

    def generate_answer(self, prompt: str) -> str:
        self.calls_count += 1
        return self.answer


def test_query_on_cache_miss_generates_and_stores_response() -> None:
    llm_provider = LocalSpyLLMProvider()
    vector_store = VectorStoreMissStub()
    orchestrator = CacheOrchestrator(
        embedder=LocalFakeEmbedder(),
        llm_provider=llm_provider,
        vector_store=vector_store,
    )

    result = orchestrator.query("What is the capital of Israel?")

    assert result.cache_hit is False
    assert result.response == "generated answer"
    assert llm_provider.calls_count == 1
    assert vector_store.insert_calls_count == 1


def test_query_on_cache_hit_returns_cached_response_without_llm_or_insert() -> None:
    llm_provider = LocalSpyLLMProvider()
    vector_store = VectorStoreHitStub(cached_response="cached answer")
    orchestrator = CacheOrchestrator(
        embedder=LocalFakeEmbedder(),
        llm_provider=llm_provider,
        vector_store=vector_store,
    )

    result = orchestrator.query("What is the capital of Israel?")

    assert result.cache_hit is True
    assert result.response == "cached answer"
    assert llm_provider.calls_count == 0
    assert vector_store.insert_calls_count == 0
