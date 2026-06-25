from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from llm_cache.config.app_config import VectorStoreConfig
from llm_cache.config.provider_options import (
    DEFAULT_EVICTION_POLICY,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_STORE_PROVIDER,
    SUPPORTED_EVICTION_POLICIES,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    normalize_provider_name,
)

DEFAULT_VECTOR_STORE_SERVER_HOST = "0.0.0.0"
DEFAULT_VECTOR_STORE_SERVER_PORT = 50052
DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS = 10
DEFAULT_VECTOR_STORE_PATH = ".cache/vector_store"
DEFAULT_VECTOR_STORE_COLLECTION = "llm_cache"
DEFAULT_VECTOR_STORE_MAX_CAPACITY = 1000


@dataclass(frozen=True)
class VectorStoreServerConfig:
    host: str
    port: int
    max_workers: int
    vector_store: VectorStoreConfig


def build_vector_store_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vector store gRPC service.")
    parser.add_argument(
        "--host",
        default=DEFAULT_VECTOR_STORE_SERVER_HOST,
        help=f"Host interface to bind. Default: {DEFAULT_VECTOR_STORE_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_VECTOR_STORE_SERVER_PORT,
        help=f"Port to listen on. Default: {DEFAULT_VECTOR_STORE_SERVER_PORT}",
    )
    parser.add_argument(
        "--vector-store-provider",
        default=DEFAULT_VECTOR_STORE_PROVIDER,
        help=f"Vector-store provider. Default: {DEFAULT_VECTOR_STORE_PROVIDER}",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold between 0 and 1. Default: {DEFAULT_SIMILARITY_THRESHOLD}",
    )
    parser.add_argument(
        "--vector-store-path",
        default=DEFAULT_VECTOR_STORE_PATH,
        help="Local persistence path for vector stores that support persistence.",
    )
    parser.add_argument(
        "--vector-store-collection",
        default=DEFAULT_VECTOR_STORE_COLLECTION,
        help="Collection name for vector stores that support named collections.",
    )
    parser.add_argument(
        "--cache-max-capacity",
        type=int,
        default=DEFAULT_VECTOR_STORE_MAX_CAPACITY,
        help=f"Maximum number of cache entries. Default: {DEFAULT_VECTOR_STORE_MAX_CAPACITY}",
    )
    parser.add_argument(
        "--eviction-policy",
        default=DEFAULT_EVICTION_POLICY,
        help=f"Cache eviction policy. Default: {DEFAULT_EVICTION_POLICY}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS,
        help=(
            "Maximum number of concurrent gRPC worker threads. "
            f"Default: {DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS}"
        ),
    )
    return parser


def parse_vector_store_server_args(
    argv: Sequence[str] | None = None,
) -> VectorStoreServerConfig:
    parser = build_vector_store_server_parser()
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    args.vector_store_provider = normalize_provider_name(args.vector_store_provider)
    args.eviction_policy = normalize_provider_name(args.eviction_policy)

    _validate_vector_store_server_runtime_args(parser, args)
    _validate_vector_store_provider(parser, args.vector_store_provider)
    _validate_eviction_policy(parser, args.eviction_policy)

    return VectorStoreServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        vector_store=VectorStoreConfig(
            provider=args.vector_store_provider,
            similarity_threshold=args.similarity_threshold,
            persist_path=args.vector_store_path,
            collection_name=args.vector_store_collection,
            max_capacity=args.cache_max_capacity,
            eviction_policy=args.eviction_policy,
        ),
    )


def _validate_vector_store_server_runtime_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.host:
        parser.error("--host must not be empty")

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    if args.cache_max_capacity < 1:
        parser.error("--cache-max-capacity must be at least 1")


def _validate_vector_store_provider(
    parser: argparse.ArgumentParser,
    vector_store_provider: str,
) -> None:
    if vector_store_provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
        parser.error(
            f"Unknown vector-store provider {vector_store_provider!r}. "
            f"Supported vector-store providers: {', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)}"
        )


def _validate_eviction_policy(
    parser: argparse.ArgumentParser,
    eviction_policy: str,
) -> None:
    if eviction_policy not in SUPPORTED_EVICTION_POLICIES:
        parser.error(
            f"Unknown eviction policy {eviction_policy!r}. "
            f"Supported eviction policies: {', '.join(SUPPORTED_EVICTION_POLICIES)}"
        )
