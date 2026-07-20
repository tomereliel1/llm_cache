from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_cache.config.app_config import ConfigError

# Top level sections supported by the shared runtime configuration file.
SECTIONS = {
    "embedding_service",
    "vector_store_service",
    "llm_service",
    "orchestrator_service",
    "web_client",
    "cli_client",
}


def load_runtime_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read and validate the structure of a runtime configuration file.

    This function validates the JSON document itself and its top level sections.
    It does not validate the individual settings inside each section, that happens when
    a section is applied to a command line parser.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        A mapping from section names to their configuration values.

    Raises:
        ConfigError: If the file cannot be read, contains invalid JSON, has an
            unsupported top level section, or contains a section that is not an
            object.
    """
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Invalid JSON in configuration file {config_path}: "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigError(f"Could not read configuration file {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError("Runtime configuration must be a JSON object")
    unknown = set(data) - SECTIONS
    if unknown:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")
    for name, section in data.items():
        if not isinstance(section, dict):
            raise ConfigError(f"Configuration section {name!r} must be a JSON object")
    return data


def load_config_section(path: str | Path, section_name: str) -> dict[str, Any]:
    """Load one required section from a runtime configuration file.

    Args:
        path: Path to the JSON configuration file.
        section_name: Name of the top-level section to retrieve, such as
            ``"embedding_service"`` or ``"web_client"``.

    Returns:
        The settings stored in the requested section.

    Raises:
        ConfigError: If the configuration is invalid or the requested section is
            missing.
    """
    data = load_runtime_config(path)
    if section_name not in data:
        raise ConfigError(f"Missing required configuration section: {section_name}")
    return data[section_name]


def config_path_from_argv(argv: Sequence[str] | None) -> str | None:
    """Extract ``--config`` from command line arguments without parsing other flags.

    Args:
        argv: Command line arguments without the executable name. If ``None``,
            arguments are read from ``sys.argv`` using ``argparse`` conventions.

    Returns:
        The value passed to ``--config``, or ``None`` when the flag is absent.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    # used because service-specific arguments are parsed later
    # by the caller's full argument parser.
    args, _ = parser.parse_known_args(argv)
    return args.config


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    section_name: str,
) -> None:
    """Use a JSON configuration section as defaults for command line arguments.

    Args:
        parser: The service or client argument parser to configure.
        argv: Command line arguments without the executable name. If ``None``,
            arguments are read from ``sys.argv``.
        section_name: Top level JSON section whose values should become defaults.

    Raises:
        SystemExit: If the file or section is invalid. The error is reported through
            ``parser.error`` so callers receive normal command line error output.
    """
    parser.add_argument("--config", help="Path to a JSON runtime configuration file.")
    path = config_path_from_argv(argv)
    if path is None:
        return
    try:
        section = load_config_section(path, section_name)
        # argparse stores every registered option as an action. Its destination is
        # the attribute name that can safely be supplied to set_defaults().
        allowed_fields = {
            action.dest for action in parser._actions if action.dest not in {"help", "config"}
        }
        unknown = set(section) - set(allowed_fields)
        if unknown:
            raise ConfigError(f"Unknown key(s) in {section_name}: {', '.join(sorted(unknown))}")
        _validate_common_values(section_name, section)
    except ConfigError as error:
        parser.error(str(error))
    # Explicit command line arguments override defaults loaded from the JSON file.
    parser.set_defaults(**section)


def _validate_int_range(
    section_name: str,
    values: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    """Validate an optional integer configuration field against a range.

    Args:
        section_name: Section name used to qualify the field in error messages.
        values: Configuration values loaded from that section.
        field: Name of the integer field to validate.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` when there is no upper bound.

    Raises:
        ConfigError: If the field exists and is not an integer in the given range.
    """
    if field not in values:
        return

    value = values[field]
    valid = (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )

    if not valid:
        expected = (
            f"an integer between {minimum} and {maximum}"
            if maximum is not None
            else f"an integer greater than or equal to {minimum}"
        )
        raise ConfigError(f"{section_name}.{field} must be {expected}")


def _validate_common_values(section_name: str, values: Mapping[str, Any]) -> None:
    """Validate configuration fields shared by multiple command line parsers.

    Args:
        section_name: Section name used to qualify fields in error messages.
        values: Configuration values loaded from that section.

    Raises:
        ConfigError: If a recognized shared field has an invalid type, is empty, or
            falls outside its permitted range.
    """
    _validate_int_range(section_name, values, "port", minimum=1, maximum=65535)
    _validate_int_range(section_name, values, "workers", minimum=1)
    _validate_int_range(section_name, values, "capacity", minimum=1)
    if "similarity_threshold" in values:
        threshold = values["similarity_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ConfigError(f"{section_name}.similarity_threshold must be a number")
        if not 0 <= threshold <= 1:
            raise ConfigError(f"{section_name}.similarity_threshold must be between 0 and 1")
    for key, value in values.items():
        if key in {"check_setup", "persistent"} and not isinstance(value, bool):
            raise ConfigError(f"{section_name}.{key} must be a boolean")
        if key in {
            "host",
            "provider",
            "model",
            "base_url",
            "path",
            "collection",
            "eviction_policy",
            "embedding_target",
            "vector_store_target",
            "llm_target",
            "orchestrator_target",
            "groq_api_key_env",
            "target",
        } and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"{section_name}.{key} must be a non-empty string")
