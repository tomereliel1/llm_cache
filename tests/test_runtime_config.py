import json
from pathlib import Path

import pytest

from llm_cache.cli.main import parse_args
from llm_cache.config.app_config import ConfigError
from llm_cache.config.embedding_server_cli_args import parse_embedding_server_args
from llm_cache.config.runtime_config import load_config_section, load_runtime_config


def write_config(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "filename",
    ["configuration_example.json", "configuration_docker_example.json"],
)
def test_example_configuration_loads(filename: str) -> None:
    path = Path(__file__).parents[1] / "configs" / filename
    config = load_runtime_config(path)
    assert set(config) == {
        "embedding_service",
        "vector_store_service",
        "llm_service",
        "orchestrator_service",
        "web_client",
        "cli_client",
    }


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_runtime_config(tmp_path / "missing.json")


def test_invalid_json_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        load_runtime_config(path)


def test_missing_section_has_clear_error(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.json", {"web_client": {}})
    with pytest.raises(ConfigError, match="Missing required configuration section"):
        load_config_section(path, "llm_service")


@pytest.mark.parametrize("port", [0, 65536, "50051"])
def test_invalid_json_port_exits(tmp_path: Path, port: object) -> None:
    path = write_config(tmp_path / "config.json", {"embedding_service": {"port": port}})
    with pytest.raises(SystemExit):
        parse_embedding_server_args(["--config", str(path)])


def test_invalid_similarity_threshold_exits(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "config.json",
        {"vector_store_service": {"similarity_threshold": 1.1}},
    )
    from llm_cache.config.vector_store_server_cli_args import parse_vector_store_server_args

    with pytest.raises(SystemExit):
        parse_vector_store_server_args(["--config", str(path)])


def test_json_key_not_registered_by_parser_exits(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "config.json",
        {"embedding_service": {"worker": 10}},
    )
    with pytest.raises(SystemExit):
        parse_embedding_server_args(["--config", str(path)])


def test_explicit_cli_flag_overrides_json(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "config.json",
        {"cli_client": {"target": "from-config:50050"}},
    )
    args = parse_args(["--config", str(path), "--target", "from-cli:60000"])
    assert args.target == "from-cli:60000"
