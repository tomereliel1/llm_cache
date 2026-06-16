from __future__ import annotations

from llm_cache.config import AppConfig
from llm_cache.config.cli_args import app_config_from_args, parse_cli_args
from llm_cache.factories import create_embedder, create_llm_provider, create_vector_store
from llm_cache.health import HealthCheckable, run_setup_check_and_exit
from llm_cache.orchestrator import CacheOrchestrator, QueryResult


def build_providers(config: AppConfig) -> tuple[HealthCheckable, HealthCheckable, HealthCheckable]:
    embedder = create_embedder(config.embedding)
    llm_provider = create_llm_provider(config.llm)
    vector_store = create_vector_store(config.vector_store)

    return embedder, llm_provider, vector_store


def build_orchestrator(config: AppConfig) -> CacheOrchestrator:
    embedder, llm_provider, vector_store = build_providers(config)

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


def main() -> None:
    args = parse_cli_args()
    config = app_config_from_args(args)
    embedder, llm_provider, vector_store = build_providers(config)

    if args.check_setup:
        run_setup_check_and_exit([embedder, llm_provider, vector_store])

    orchestrator = CacheOrchestrator(
        embedder=embedder,
        llm_provider=llm_provider,
        vector_store=vector_store,
    )
    result = run_query(orchestrator, args.prompt)
    print_result(args.prompt, result)


if __name__ == "__main__":
    main()
