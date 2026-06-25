from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from llm_cache.config.app_config import LLMConfig
from llm_cache.config.provider_options import (
    DEFAULT_LLM_PROVIDER,
    SUPPORTED_LLM_PROVIDERS,
    default_llm_model,
    normalize_provider_name,
)

DEFAULT_LLM_SERVER_HOST = "0.0.0.0"
DEFAULT_LLM_SERVER_PORT = 50053
DEFAULT_LLM_SERVER_MAX_WORKERS = 10


@dataclass(frozen=True)
class LLMServerConfig:
    host: str
    port: int
    max_workers: int
    check_setup: bool
    llm: LLMConfig


def build_llm_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM gRPC service.")
    parser.add_argument(
        "--host",
        default=DEFAULT_LLM_SERVER_HOST,
        help=f"Host interface to bind. Default: {DEFAULT_LLM_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_LLM_SERVER_PORT,
        help=f"Port to listen on. Default: {DEFAULT_LLM_SERVER_PORT}",
    )
    parser.add_argument(
        "--llm-provider",
        default=DEFAULT_LLM_PROVIDER,
        help=f"LLM provider. Default: {DEFAULT_LLM_PROVIDER}",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name. If omitted, the selected provider's default is used.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_LLM_SERVER_MAX_WORKERS,
        help=(
            "Maximum number of concurrent gRPC worker threads. "
            f"Default: {DEFAULT_LLM_SERVER_MAX_WORKERS}"
        ),
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Run provider health checks before starting the server.",
    )
    return parser


def parse_llm_server_args(argv: Sequence[str] | None = None) -> LLMServerConfig:
    parser = build_llm_server_parser()
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    args.llm_provider = normalize_provider_name(args.llm_provider)

    _validate_llm_server_runtime_args(parser, args)
    _validate_llm_provider(parser, args.llm_provider)

    if args.llm_model is None:
        args.llm_model = default_llm_model(args.llm_provider)
    else:
        args.llm_model = args.llm_model.strip()

    if args.llm_api_key_env is not None:
        args.llm_api_key_env = args.llm_api_key_env.strip()
        if not args.llm_api_key_env:
            parser.error("--llm-api-key-env must not be empty")

    _validate_llm_model(parser, args.llm_provider, args.llm_model)

    return LLMServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        check_setup=args.check_setup,
        llm=LLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
        ),
    )


def _validate_llm_server_runtime_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.host:
        parser.error("--host must not be empty")

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")


def _validate_llm_provider(
    parser: argparse.ArgumentParser,
    llm_provider: str,
) -> None:
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        parser.error(
            f"Unknown LLM provider {llm_provider!r}. "
            f"Supported LLM providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
        )


def _validate_llm_model(
    parser: argparse.ArgumentParser,
    llm_provider: str,
    llm_model: str,
) -> None:
    provider_option = SUPPORTED_LLM_PROVIDERS[llm_provider]
    if llm_model not in provider_option.supported_models:
        parser.error(
            f"Unsupported LLM model {llm_model!r} "
            f"for provider {llm_provider!r}. "
            f"Supported models: {', '.join(provider_option.supported_models)}"
        )
