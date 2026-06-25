import pytest

from llm_cache.config.embedding_server_cli_args import (
    DEFAULT_EMBEDDING_SERVER_HOST,
    DEFAULT_EMBEDDING_SERVER_MAX_WORKERS,
    DEFAULT_EMBEDDING_SERVER_PORT,
    parse_embedding_server_args,
)
from llm_cache.config.provider_options import (
    DEFAULT_EMBEDDING_PROVIDER,
    SUPPORTED_EMBEDDING_PROVIDERS,
    default_embedding_model,
)


def test_parse_embedding_server_args_uses_shared_defaults() -> None:
    config = parse_embedding_server_args([])

    assert config.host == DEFAULT_EMBEDDING_SERVER_HOST
    assert config.port == DEFAULT_EMBEDDING_SERVER_PORT
    assert config.max_workers == DEFAULT_EMBEDDING_SERVER_MAX_WORKERS
    assert config.check_setup is False
    assert config.embedding.provider == DEFAULT_EMBEDDING_PROVIDER
    assert config.embedding.model == default_embedding_model(DEFAULT_EMBEDDING_PROVIDER)


def test_parse_embedding_server_args_accepts_explicit_values() -> None:
    provider = DEFAULT_EMBEDDING_PROVIDER
    model = SUPPORTED_EMBEDDING_PROVIDERS[provider].supported_models[0]

    config = parse_embedding_server_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "60000",
            "--embedding-provider",
            provider,
            "--embedding-model",
            model,
            "--max-workers",
            "3",
            "--check-setup",
        ]
    )

    assert config.host == "127.0.0.1"
    assert config.port == 60000
    assert config.max_workers == 3
    assert config.check_setup is True
    assert config.embedding.provider == provider
    assert config.embedding.model == model


def test_parse_embedding_server_args_normalizes_provider_name() -> None:
    config = parse_embedding_server_args(["--embedding-provider", " Ollama "])

    assert config.embedding.provider == "ollama"
    assert config.embedding.model == default_embedding_model("ollama")


def test_invalid_embedding_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_embedding_server_args(["--embedding-provider", "bad-provider"])


def test_invalid_embedding_model_for_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_embedding_server_args(["--embedding-model", "bad-model"])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--host", ""),
        ("--host", "   "),
        ("--port", "0"),
        ("--port", "65536"),
        ("--max-workers", "0"),
    ],
)
def test_invalid_embedding_server_runtime_args_exit(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_embedding_server_args([flag, value])
