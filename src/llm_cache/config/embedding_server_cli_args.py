from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from llm_cache.config.app_config import EmbeddingConfig
from llm_cache.config.provider_options import (
    DEFAULT_EMBEDDING_PROVIDER,
    SUPPORTED_EMBEDDING_PROVIDERS,
    default_embedding_model,
    normalize_provider_name,
)
from llm_cache.config.runtime_config import apply_config_defaults

DEFAULT_EMBEDDING_SERVER_HOST = "0.0.0.0"
DEFAULT_EMBEDDING_SERVER_PORT = 50051
DEFAULT_EMBEDDING_SERVER_MAX_WORKERS = 10


@dataclass(frozen=True)
class EmbeddingServerConfig:
    host: str
    port: int
    max_workers: int
    check_setup: bool
    embedding: EmbeddingConfig


def build_embedding_server_parser(
    argv: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embedding gRPC service.")
    parser.add_argument(
        "--host",
        default=DEFAULT_EMBEDDING_SERVER_HOST,
        help=f"Host interface to bind. Default: {DEFAULT_EMBEDDING_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_EMBEDDING_SERVER_PORT,
        help=f"Port to listen on. Default: {DEFAULT_EMBEDDING_SERVER_PORT}",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider. Default: {DEFAULT_EMBEDDING_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model name. If omitted, the default model for the selected "
            "embedding provider is used."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EMBEDDING_SERVER_MAX_WORKERS,
        help=(
            "Maximum number of concurrent gRPC worker threads. "
            f"Default: {DEFAULT_EMBEDDING_SERVER_MAX_WORKERS}"
        ),
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Run provider health checks before starting the server.",
    )
    parser.add_argument("--base-url", default=None, help="Optional provider API base URL.")
    apply_config_defaults(
        parser,
        argv,
        "embedding_service",
    )
    return parser


def parse_embedding_server_args(
    argv: Sequence[str] | None = None,
) -> EmbeddingServerConfig:
    parser = build_embedding_server_parser(argv)
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    args.provider = normalize_provider_name(args.provider)

    _validate_embedding_server_runtime_args(parser, args)
    _validate_embedding_provider(parser, args.provider)

    if args.model is None:
        args.model = default_embedding_model(args.provider)
    else:
        args.model = args.model.strip()

    _validate_embedding_model(parser, args.provider, args.model)

    return EmbeddingServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        check_setup=args.check_setup,
        embedding=EmbeddingConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
        ),
    )


def _validate_embedding_server_runtime_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.host:
        parser.error("--host must not be empty")

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    if args.workers < 1:
        parser.error("--workers must be at least 1")


def _validate_embedding_provider(
    parser: argparse.ArgumentParser,
    embedding_provider: str,
) -> None:
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        parser.error(
            f"Unknown embedding provider {embedding_provider!r}. "
            f"Supported embedding providers: {', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}"
        )


def _validate_embedding_model(
    parser: argparse.ArgumentParser,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    embedding_option = SUPPORTED_EMBEDDING_PROVIDERS[embedding_provider]
    if embedding_model not in embedding_option.supported_models:
        parser.error(
            f"Unsupported embedding model {embedding_model!r} "
            f"for provider {embedding_provider!r}. "
            f"Supported models: {', '.join(embedding_option.supported_models)}"
        )
