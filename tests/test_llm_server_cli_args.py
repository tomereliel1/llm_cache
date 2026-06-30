import pytest

from llm_cache.config.llm_server_cli_args import (
    DEFAULT_LLM_SERVER_HOST,
    DEFAULT_LLM_SERVER_MAX_WORKERS,
    DEFAULT_LLM_SERVER_PORT,
    parse_llm_server_args,
)
from llm_cache.config.provider_options import (
    DEFAULT_LLM_PROVIDER,
    SUPPORTED_LLM_PROVIDERS,
    default_llm_model,
)


def test_parse_llm_server_args_uses_shared_defaults() -> None:
    config = parse_llm_server_args([])

    assert config.host == DEFAULT_LLM_SERVER_HOST
    assert config.port == DEFAULT_LLM_SERVER_PORT
    assert config.max_workers == DEFAULT_LLM_SERVER_MAX_WORKERS
    assert config.check_setup is False
    assert config.llm.provider == DEFAULT_LLM_PROVIDER
    assert config.llm.model == default_llm_model(DEFAULT_LLM_PROVIDER)
    assert config.llm.api_key_env is None


@pytest.mark.parametrize("provider", SUPPORTED_LLM_PROVIDERS)
def test_llm_provider_uses_provider_specific_default_model(provider: str) -> None:
    config = parse_llm_server_args(["--llm-provider", provider])

    assert config.llm.provider == provider
    assert config.llm.model == default_llm_model(provider)


def test_parse_llm_server_args_accepts_explicit_values() -> None:
    config = parse_llm_server_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "60001",
            "--llm-provider",
            "groq",
            "--llm-model",
            "llama-3.1-8b-instant",
            "--llm-api-key-env",
            "CUSTOM_GROQ_KEY",
            "--max-workers",
            "4",
            "--check-setup",
        ]
    )

    assert config.host == "127.0.0.1"
    assert config.port == 60001
    assert config.max_workers == 4
    assert config.check_setup is True
    assert config.llm.provider == "groq"
    assert config.llm.model == "llama-3.1-8b-instant"
    assert config.llm.api_key_env == "CUSTOM_GROQ_KEY"


def test_parse_llm_server_args_normalizes_provider_name() -> None:
    config = parse_llm_server_args(["--llm-provider", " Groq "])

    assert config.llm.provider == "groq"
    assert config.llm.model == default_llm_model("groq")


def test_invalid_llm_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_llm_server_args(["--llm-provider", "bad-provider"])


def test_invalid_llm_model_for_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_llm_server_args(
            [
                "--llm-provider",
                "ollama",
                "--llm-model",
                "llama-3.1-8b-instant",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--host", ""),
        ("--host", "   "),
        ("--port", "0"),
        ("--port", "65536"),
        ("--max-workers", "0"),
        ("--llm-api-key-env", ""),
        ("--llm-api-key-env", "   "),
    ],
)
def test_invalid_llm_server_args_exit(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_llm_server_args([flag, value])
