from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from llm_cache.config.app_config import VectorStoreConfig
from llm_cache.config.provider_options import (
    DEFAULT_CHROMA_DISTANCE_FUNCTION,
    DEFAULT_EVICTION_POLICY,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_STORE_PROVIDER,
    SUPPORTED_CHROMA_DISTANCE_FUNCTIONS,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    normalize_provider_name,
)
from llm_cache.config.runtime_config import apply_config_defaults

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
    check_setup: bool
    vector_store: VectorStoreConfig


def build_vector_store_server_parser(
    argv: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
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
        "--provider",
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
        "--path",
        default=DEFAULT_VECTOR_STORE_PATH,
        help="Local persistence path for vector stores when persistence is enabled.",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Persist vector-store data between launches. Chroma is ephemeral by default.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_STORE_COLLECTION,
        help="Collection name for vector stores that support named collections.",
    )
    parser.add_argument(
        "--distance-function",
        choices=SUPPORTED_CHROMA_DISTANCE_FUNCTIONS,
        default=DEFAULT_CHROMA_DISTANCE_FUNCTION,
        help=(
            "Distance function for Chroma vector stores. Supported functions: "
            f"{', '.join(SUPPORTED_CHROMA_DISTANCE_FUNCTIONS)}. "
            f"Default: {DEFAULT_CHROMA_DISTANCE_FUNCTION}"
        ),
    )
    parser.add_argument(
        "--capacity",
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
        "--workers",
        type=int,
        default=DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS,
        help=(
            "Maximum number of concurrent gRPC worker threads. "
            f"Default: {DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS}"
        ),
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Run provider health checks before starting the server.",
    )
    apply_config_defaults(
        parser,
        argv,
        "vector_store_service",
    )
    return parser


def parse_vector_store_server_args(
    argv: Sequence[str] | None = None,
) -> VectorStoreServerConfig:
    parser = build_vector_store_server_parser(argv)
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    args.provider = normalize_provider_name(args.provider)
    args.eviction_policy = normalize_provider_name(args.eviction_policy)
    args.distance_function = normalize_provider_name(args.distance_function)

    _validate_vector_store_server_runtime_args(parser, args)
    _validate_vector_store_provider(parser, args.provider)
    _validate_distance_function(parser, args.distance_function)

    return VectorStoreServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        check_setup=args.check_setup,
        vector_store=VectorStoreConfig(
            provider=args.provider,
            similarity_threshold=args.similarity_threshold,
            persist_path=args.path,
            collection_name=args.collection,
            max_capacity=args.capacity,
            eviction_policy=args.eviction_policy,
            persistent=args.persistent,
            distance_function=args.distance_function,
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

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.capacity < 1:
        parser.error("--capacity must be at least 1")


def _validate_vector_store_provider(
    parser: argparse.ArgumentParser,
    vector_store_provider: str,
) -> None:
    if vector_store_provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
        parser.error(
            f"Unknown vector-store provider {vector_store_provider!r}. "
            f"Supported vector-store providers: {', '.join(SUPPORTED_VECTOR_STORE_PROVIDERS)}"
        )


def _validate_distance_function(
    parser: argparse.ArgumentParser,
    distance_function: str,
) -> None:
    if distance_function not in SUPPORTED_CHROMA_DISTANCE_FUNCTIONS:
        parser.error(
            f"Unsupported distance function {distance_function!r}. "
            f"Supported distance functions: {', '.join(SUPPORTED_CHROMA_DISTANCE_FUNCTIONS)}"
        )
