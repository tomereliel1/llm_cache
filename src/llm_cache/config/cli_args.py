from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_cache.config.app_config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from llm_cache.config.provider_options import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_PROMPT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_STORE_PROVIDER,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    default_embedding_model,
    default_llm_model,
    format_supported_configs,
    normalize_provider_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local LLM semantic cache demo.",
        epilog=(
            "Example:\n"
            "  uv run python main.py "
            '--prompt "What is the capital of Israel?" '
            "--embedding-provider ollama "
            "--embedding-model embeddinggemma "
            "--llm-provider ollama "
            "--vector-store-provider vector-store-miss-stub "
            "--similarity-threshold 0.85\n\n"
            "For provider/model details, run:\n"
            "  uv run python main.py --list-supported-configs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt to send to the cache system. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=tuple(SUPPORTED_EMBEDDING_PROVIDERS),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider. Default: {DEFAULT_EMBEDDING_PROVIDER}",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Embedding model name. If omitted, the default model for the "
            "selected embedding provider is used."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=tuple(SUPPORTED_LLM_PROVIDERS),
        default=DEFAULT_LLM_PROVIDER,
        help=f"LLM provider. Default: {DEFAULT_LLM_PROVIDER}",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=(
            "LLM model name. If omitted, the default model for the selected LLM provider is used."
        ),
    )
    parser.add_argument(
        "--vector-store-provider",
        type=str,
        choices=tuple(SUPPORTED_VECTOR_STORE_PROVIDERS),
        default=DEFAULT_VECTOR_STORE_PROVIDER,
        help=f"Vector-store provider. Default: {DEFAULT_VECTOR_STORE_PROVIDER}",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(f"Similarity threshold between 0 and 1. Default: {DEFAULT_SIMILARITY_THRESHOLD}"),
    )
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default=".cache/vector_store",
        help="Local persistence path for vector stores that support persistence.",
    )
    parser.add_argument(
        "--vector-store-collection",
        type=str,
        default="llm_cache",
        help="Collection name for vector stores that support named collections.",
    )
    parser.add_argument(
        "--cache-max-capacity",
        type=int,
        default=1000,
        help="Maximum number of entries the vector cache can store. Default: 1000",
    )
    parser.add_argument(
        "--list-supported-configs",
        action="store_true",
        help="Print supported providers, models, and defaults, then exit.",
    )

    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_supported_configs:
        print(format_supported_configs())
        raise SystemExit(0)

    args.embedding_provider = normalize_provider_name(args.embedding_provider)
    args.llm_provider = normalize_provider_name(args.llm_provider)
    args.vector_store_provider = normalize_provider_name(args.vector_store_provider)

    if args.embedding_model is None:
        args.embedding_model = default_embedding_model(args.embedding_provider)
    else:
        args.embedding_model = args.embedding_model.strip()

    if args.llm_model is None:
        args.llm_model = default_llm_model(args.llm_provider)
    else:
        args.llm_model = args.llm_model.strip()

    _validate_args(parser, args)
    return args


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")

    if not 0 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be between 0 and 1")

    if not args.vector_store_path.strip():
        parser.error("--vector-store-path must not be empty")

    if not args.vector_store_collection.strip():
        parser.error("--vector-store-collection must not be empty")

    if args.cache_max_capacity < 1:
        parser.error("--cache-max-capacity must be at least 1")

    embedding_option = SUPPORTED_EMBEDDING_PROVIDERS[args.embedding_provider]
    if args.embedding_model not in embedding_option.supported_models:
        parser.error(
            f"Unsupported embedding model {args.embedding_model!r} "
            f"for provider {args.embedding_provider!r}. "
            f"Supported models: {', '.join(embedding_option.supported_models)}"
        )

    llm_option = SUPPORTED_LLM_PROVIDERS[args.llm_provider]
    if args.llm_model not in llm_option.supported_models:
        parser.error(
            f"Unsupported LLM model {args.llm_model!r} "
            f"for provider {args.llm_provider!r}. "
            f"Supported models: {', '.join(llm_option.supported_models)}"
        )


def app_config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(
            provider=args.embedding_provider,
            model=args.embedding_model,
        ),
        llm=LLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
        ),
        vector_store=VectorStoreConfig(
            provider=args.vector_store_provider,
            similarity_threshold=args.similarity_threshold,
            persist_path=args.vector_store_path,
            collection_name=args.vector_store_collection,
            max_capacity=args.cache_max_capacity,
        ),
    )
