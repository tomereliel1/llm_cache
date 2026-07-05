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
from llm_cache.config.runtime_config import apply_config_defaults

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


def build_llm_server_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
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
        "--provider",
        default=DEFAULT_LLM_PROVIDER,
        help=f"LLM provider. Default: {DEFAULT_LLM_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name. If omitted, the selected provider's default is used.",
    )
    parser.add_argument(
        "--groq-api-key-env",
        default=None,
        help=(
            "Name of the environment variable containing the Groq API key; "
            "the key itself must not be passed here. Default: GROQ_API_KEY"
        ),
    )
    parser.add_argument(
        "--workers",
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
    parser.add_argument("--base-url", default=None, help="Optional provider API base URL.")
    apply_config_defaults(
        parser,
        argv,
        "llm_service",
    )
    return parser


def parse_llm_server_args(argv: Sequence[str] | None = None) -> LLMServerConfig:
    parser = build_llm_server_parser(argv)
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    args.provider = normalize_provider_name(args.provider)

    _validate_llm_server_runtime_args(parser, args)
    _validate_llm_provider(parser, args.provider)

    if args.model is None:
        args.model = default_llm_model(args.provider)
    else:
        args.model = args.model.strip()

    if args.groq_api_key_env is not None:
        args.groq_api_key_env = args.groq_api_key_env.strip()
        if not args.groq_api_key_env:
            parser.error("--groq-api-key-env must not be empty")

    _validate_llm_model(parser, args.provider, args.model)

    return LLMServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        check_setup=args.check_setup,
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            groq_api_key_env=args.groq_api_key_env,
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

    if args.workers < 1:
        parser.error("--workers must be at least 1")


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
