import pytest

from llm_cache.config.cli_args import app_config_from_args, parse_cli_args
from llm_cache.config.provider_options import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EVICTION_POLICY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_STORE_PROVIDER,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_EVICTION_POLICIES,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    default_embedding_model,
    default_llm_model,
)


def test_parse_cli_args_uses_registry_defaults() -> None:
    args = parse_cli_args([])

    assert args.embedding_provider == DEFAULT_EMBEDDING_PROVIDER
    assert args.embedding_model == default_embedding_model(DEFAULT_EMBEDDING_PROVIDER)
    assert args.llm_provider == DEFAULT_LLM_PROVIDER
    assert args.llm_model == default_llm_model(DEFAULT_LLM_PROVIDER)
    assert args.vector_store_provider == DEFAULT_VECTOR_STORE_PROVIDER
    assert args.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert args.eviction_policy == DEFAULT_EVICTION_POLICY


@pytest.mark.parametrize("provider", SUPPORTED_LLM_PROVIDERS.keys())
def test_llm_provider_uses_provider_specific_default_model(provider: str) -> None:
    args = parse_cli_args(["--llm-provider", provider])

    assert args.llm_provider == provider
    assert args.llm_model == default_llm_model(provider)


@pytest.mark.parametrize("provider", SUPPORTED_EMBEDDING_PROVIDERS.keys())
def test_embedding_provider_uses_provider_specific_default_model(provider: str) -> None:
    args = parse_cli_args(["--embedding-provider", provider])

    assert args.embedding_provider == provider
    assert args.embedding_model == default_embedding_model(provider)


def test_registry_defaults_are_valid() -> None:
    assert DEFAULT_EMBEDDING_PROVIDER in SUPPORTED_EMBEDDING_PROVIDERS
    assert DEFAULT_LLM_PROVIDER in SUPPORTED_LLM_PROVIDERS
    assert DEFAULT_VECTOR_STORE_PROVIDER in SUPPORTED_VECTOR_STORE_PROVIDERS

    embedding_default_model = default_embedding_model(DEFAULT_EMBEDDING_PROVIDER)
    assert (
        embedding_default_model
        in SUPPORTED_EMBEDDING_PROVIDERS[DEFAULT_EMBEDDING_PROVIDER].supported_models
    )

    llm_default_model = default_llm_model(DEFAULT_LLM_PROVIDER)
    assert llm_default_model in SUPPORTED_LLM_PROVIDERS[DEFAULT_LLM_PROVIDER].supported_models


def test_explicit_llm_model_override_works() -> None:
    provider = DEFAULT_LLM_PROVIDER
    model = SUPPORTED_LLM_PROVIDERS[provider].supported_models[0]

    args = parse_cli_args(["--llm-provider", provider, "--llm-model", model])

    assert args.llm_provider == provider
    assert args.llm_model == model


def test_app_config_from_args_uses_parsed_values() -> None:
    args = parse_cli_args([])
    config = app_config_from_args(args)

    assert config.embedding.provider == args.embedding_provider
    assert config.embedding.model == args.embedding_model
    assert config.llm.provider == args.llm_provider
    assert config.llm.model == args.llm_model
    assert config.vector_store.provider == args.vector_store_provider
    assert config.vector_store.similarity_threshold == args.similarity_threshold
    assert config.vector_store.persist_path == args.vector_store_path
    assert config.vector_store.collection_name == args.vector_store_collection
    assert config.vector_store.max_capacity == args.cache_max_capacity
    assert config.vector_store.eviction_policy == args.eviction_policy


def test_app_config_from_args_uses_vector_store_path_and_collection() -> None:
    args = parse_cli_args(
        [
            "--vector-store-provider",
            "chroma",
            "--vector-store-path",
            ".cache/custom",
            "--vector-store-collection",
            "custom_collection",
            "--cache-max-capacity",
            "42",
            "--eviction-policy",
            "lru",
        ]
    )
    config = app_config_from_args(args)

    assert config.vector_store.persist_path == ".cache/custom"
    assert config.vector_store.collection_name == "custom_collection"
    assert config.vector_store.max_capacity == 42
    assert config.vector_store.eviction_policy == "lru"


@pytest.mark.parametrize("policy", SUPPORTED_EVICTION_POLICIES.keys())
def test_eviction_policy_accepts_supported_policy(policy: str) -> None:
    args = parse_cli_args(["--eviction-policy", policy])

    assert args.eviction_policy == policy


@pytest.mark.parametrize("threshold", ["-0.1", "1.1"])
def test_invalid_similarity_threshold_exits(threshold: str) -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--similarity-threshold", threshold])


def test_empty_prompt_exits() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--prompt", ""])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--vector-store-path", ""),
        ("--vector-store-collection", ""),
    ],
)
def test_empty_vector_store_config_value_exits(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_cli_args([flag, value])


def test_invalid_cache_max_capacity_exits() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--cache-max-capacity", "0"])


def test_invalid_llm_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--llm-provider", "bad-provider"])


def test_invalid_eviction_policy_exits() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--eviction-policy", "bad-policy"])


def test_invalid_llm_model_for_provider_exits() -> None:
    providers = list(SUPPORTED_LLM_PROVIDERS)
    if len(providers) < 2:
        pytest.skip("Need at least two LLM providers for this test")

    provider_a = providers[0]
    provider_b = providers[1]
    invalid_model = SUPPORTED_LLM_PROVIDERS[provider_b].supported_models[0]

    if invalid_model in SUPPORTED_LLM_PROVIDERS[provider_a].supported_models:
        pytest.skip("Providers share the same model; no invalid combination to test")

    with pytest.raises(SystemExit):
        parse_cli_args(["--llm-provider", provider_a, "--llm-model", invalid_model])


def test_list_supported_configs_prints_registered_options(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_args(["--list-supported-configs"])

    assert exc_info.value.code == 0

    output = capsys.readouterr().out

    for provider in SUPPORTED_EMBEDDING_PROVIDERS:
        assert provider in output

    for provider in SUPPORTED_LLM_PROVIDERS:
        assert provider in output

    for provider in SUPPORTED_VECTOR_STORE_PROVIDERS:
        assert provider in output
