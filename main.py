from llm_cache.config import AppConfig, ConfigError, EmbeddingConfig, LLMConfig, VectorStoreConfig
from llm_cache.factories import create_embedder, create_llm_provider, create_vector_store
from llm_cache.orchestrator import CacheOrchestrator, QueryResult

DEFAULT_DEMO_PROMPT = "Explain semantic caching in one sentence."
EMBEDDING_MODEL_NAME = "embeddinggemma"
LLM_MODEL_NAME = "llama3.2:3b"
SIMILARITY_THRESHOLD = 0.8
VECTOR_STORE_PROVIDER = "vector-store-miss-stub"


def build_demo_config() -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(
            provider="ollama",
            model=EMBEDDING_MODEL_NAME,
        ),
        llm=LLMConfig(
            provider="ollama",
            model=LLM_MODEL_NAME,
        ),
        vector_store=VectorStoreConfig(
            provider=VECTOR_STORE_PROVIDER,
        ),
        similarity_threshold=SIMILARITY_THRESHOLD,
    )


def build_orchestrator(config: AppConfig) -> CacheOrchestrator:
    embedder = create_embedder(config.embedding)
    llm_provider = create_llm_provider(config.llm)
    vector_store = create_vector_store(config.vector_store)

    return CacheOrchestrator(
        embedder=embedder,
        llm_provider=llm_provider,
        vector_store=vector_store,
        similarity_threshold=config.similarity_threshold,
    )


def run_demo_query(orchestrator: CacheOrchestrator, prompt: str) -> QueryResult:
    return orchestrator.query(prompt)


def main() -> None:
    try:
        config = build_demo_config()
        orchestrator = build_orchestrator(config)
        result = run_demo_query(orchestrator, DEFAULT_DEMO_PROMPT)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    print("Prompt:", DEFAULT_DEMO_PROMPT)
    print("Response:", result.response)
    print("Cache hit:", result.cache_hit)


if __name__ == "__main__":
    main()
