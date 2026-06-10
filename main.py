from __future__ import annotations

from llm_cache.config import AppConfig
from llm_cache.config.cli_args import app_config_from_args, parse_cli_args
from llm_cache.factories import create_embedder, create_llm_provider, create_vector_store
from llm_cache.orchestrator import CacheOrchestrator, QueryResult


def build_orchestrator(config: AppConfig) -> CacheOrchestrator:
    embedder = create_embedder(config.embedding)
    llm_provider = create_llm_provider(config.llm)
    vector_store = create_vector_store(config.vector_store)

    return CacheOrchestrator(
        embedder=embedder,
        llm_provider=llm_provider,
        vector_store=vector_store,
    )


def run_query(orchestrator: CacheOrchestrator, prompt: str) -> QueryResult:
    return orchestrator.query(prompt)


def print_result(prompt: str, result: QueryResult) -> None:
    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Response:")
    print(result.response)

    print()
    print("Metadata:")
    print(f"cache_hit: {result.cache_hit}")

    if hasattr(result, "similarity_score"):
        print(f"similarity_score: {result.similarity_score}")


def main() -> None:
    args = parse_cli_args()
    config = app_config_from_args(args)
    orchestrator = build_orchestrator(config)
    result = run_query(orchestrator, args.prompt)
    print_result(args.prompt, result)


if __name__ == "__main__":
    main()
