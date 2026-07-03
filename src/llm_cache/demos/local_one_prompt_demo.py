from __future__ import annotations

import sys

from llm_cache.config import AppConfig, ConfigError
from llm_cache.config.cli_args import app_config_from_args, parse_cli_args
from llm_cache.factories import create_embedder, create_llm_provider, create_vector_store
from llm_cache.health import HealthCheckable, run_setup_check_and_exit
from llm_cache.orchestrator import CacheOrchestrator, QueryResult


def build_providers(config: AppConfig) -> tuple[HealthCheckable, HealthCheckable, HealthCheckable]:
    return (
        create_embedder(config.embedding),
        create_llm_provider(config.llm),
        create_vector_store(config.vector_store),
    )


def build_orchestrator(config: AppConfig) -> CacheOrchestrator:
    embedder, llm_provider, vector_store = build_providers(config)
    return CacheOrchestrator(embedder, llm_provider, vector_store)


def run_query(orchestrator: CacheOrchestrator, prompt: str) -> QueryResult:
    return orchestrator.query(prompt)


def print_result(prompt: str, result: QueryResult) -> None:
    print(f"\nPrompt:\n{prompt}")
    print(f"\nResponse:\n{result.response}")
    print(f"\nMetadata:\ncache_hit: {result.cache_hit}")


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    try:
        config = app_config_from_args(args)
        embedder, llm_provider, vector_store = build_providers(config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    if args.check_setup:
        run_setup_check_and_exit([embedder, llm_provider, vector_store])
    orchestrator = CacheOrchestrator(embedder, llm_provider, vector_store)
    print_result(args.prompt, run_query(orchestrator, args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
