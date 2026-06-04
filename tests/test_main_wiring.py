import main
from llm_cache.config import AppConfig
from llm_cache.orchestrator import CacheOrchestrator, QueryResult
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy, VectorStoreMissStub


def test_build_demo_config_returns_expected_default_config() -> None:
    config = main.build_demo_config()

    assert isinstance(config, AppConfig)
    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "embeddinggemma"
    assert config.llm.provider == "ollama"
    assert config.llm.model == "llama3.2:3b"
    assert config.vector_store.provider == "vector-store-miss-stub"
    assert 0 <= config.similarity_threshold <= 1


def test_build_orchestrator_uses_factories(monkeypatch) -> None:
    config = main.build_demo_config()
    embedder = EmbedderStub()
    llm_provider = LLMProviderSpy()
    vector_store = VectorStoreMissStub()
    factory_calls = []

    def create_embedder(config):
        factory_calls.append(("embedding", config))
        return embedder

    def create_llm_provider(config):
        factory_calls.append(("llm", config))
        return llm_provider

    def create_vector_store(config):
        factory_calls.append(("vector_store", config))
        return vector_store

    monkeypatch.setattr(main, "create_embedder", create_embedder)
    monkeypatch.setattr(main, "create_llm_provider", create_llm_provider)
    monkeypatch.setattr(main, "create_vector_store", create_vector_store)

    orchestrator = main.build_orchestrator(config)

    assert isinstance(orchestrator, CacheOrchestrator)
    assert factory_calls == [
        ("embedding", config.embedding),
        ("llm", config.llm),
        ("vector_store", config.vector_store),
    ]


def test_run_demo_query_returns_query_result() -> None:
    orchestrator = CacheOrchestrator(
        embedder=EmbedderStub(),
        llm_provider=LLMProviderSpy(),
        vector_store=VectorStoreMissStub(),
        similarity_threshold=0.8,
    )

    result = main.run_demo_query(orchestrator, "What is semantic caching?")

    assert result == QueryResult(response="generated answer", cache_hit=False)
