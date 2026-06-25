import pytest

from llm_cache.config.provider_options import (
    DEFAULT_EVICTION_POLICY,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_STORE_PROVIDER,
)
from llm_cache.config.vector_store_server_cli_args import (
    DEFAULT_VECTOR_STORE_COLLECTION,
    DEFAULT_VECTOR_STORE_MAX_CAPACITY,
    DEFAULT_VECTOR_STORE_PATH,
    DEFAULT_VECTOR_STORE_SERVER_HOST,
    DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS,
    DEFAULT_VECTOR_STORE_SERVER_PORT,
    parse_vector_store_server_args,
)


def test_parse_vector_store_server_args_uses_shared_defaults() -> None:
    config = parse_vector_store_server_args([])

    assert config.host == DEFAULT_VECTOR_STORE_SERVER_HOST
    assert config.port == DEFAULT_VECTOR_STORE_SERVER_PORT
    assert config.max_workers == DEFAULT_VECTOR_STORE_SERVER_MAX_WORKERS
    assert config.vector_store.provider == DEFAULT_VECTOR_STORE_PROVIDER
    assert config.vector_store.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert config.vector_store.persist_path == DEFAULT_VECTOR_STORE_PATH
    assert config.vector_store.collection_name == DEFAULT_VECTOR_STORE_COLLECTION
    assert config.vector_store.max_capacity == DEFAULT_VECTOR_STORE_MAX_CAPACITY
    assert config.vector_store.eviction_policy == DEFAULT_EVICTION_POLICY


def test_parse_vector_store_server_args_accepts_explicit_values() -> None:
    config = parse_vector_store_server_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "60000",
            "--vector-store-provider",
            "in-memory",
            "--similarity-threshold",
            "0.9",
            "--vector-store-path",
            ".cache/custom",
            "--vector-store-collection",
            "custom_collection",
            "--cache-max-capacity",
            "42",
            "--eviction-policy",
            "lru",
            "--max-workers",
            "3",
        ]
    )

    assert config.host == "127.0.0.1"
    assert config.port == 60000
    assert config.max_workers == 3
    assert config.vector_store.provider == "in-memory"
    assert config.vector_store.similarity_threshold == 0.9
    assert config.vector_store.persist_path == ".cache/custom"
    assert config.vector_store.collection_name == "custom_collection"
    assert config.vector_store.max_capacity == 42
    assert config.vector_store.eviction_policy == "lru"


def test_parse_vector_store_server_args_normalizes_names() -> None:
    config = parse_vector_store_server_args(
        [
            "--vector-store-provider",
            " In-Memory ",
            "--eviction-policy",
            " LRU ",
        ]
    )

    assert config.vector_store.provider == "in-memory"
    assert config.vector_store.eviction_policy == "lru"


def test_invalid_vector_store_provider_exits() -> None:
    with pytest.raises(SystemExit):
        parse_vector_store_server_args(["--vector-store-provider", "bad-provider"])


def test_invalid_eviction_policy_exits() -> None:
    with pytest.raises(SystemExit):
        parse_vector_store_server_args(["--eviction-policy", "bad-policy"])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--host", ""),
        ("--host", "   "),
        ("--port", "0"),
        ("--port", "65536"),
        ("--max-workers", "0"),
        ("--cache-max-capacity", "0"),
    ],
)
def test_invalid_vector_store_server_runtime_args_exit(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_vector_store_server_args([flag, value])
