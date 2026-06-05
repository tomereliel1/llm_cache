import main
from llm_cache.config.cli_args import app_config_from_args, parse_cli_args
from llm_cache.orchestrator import CacheOrchestrator, QueryResult
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy, VectorStoreMissStub


def test_build_orchestrator_uses_factories(monkeypatch) -> None:
    config = app_config_from_args(parse_cli_args([]))
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


def test_run_query_returns_query_result() -> None:
    orchestrator = CacheOrchestrator(
        embedder=EmbedderStub(),
        llm_provider=LLMProviderSpy(),
        vector_store=VectorStoreMissStub(),
        similarity_threshold=0.8,
    )

    result = main.run_query(orchestrator, "What is semantic caching?")

    assert result == QueryResult(response="generated answer", cache_hit=False)
