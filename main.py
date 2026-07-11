"""Convenient entrypoint for the orchestrator clients."""

from __future__ import annotations

import argparse

MODES = {"web", "cli"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an LLM cache client. Defaults to the web client.",
        add_help=False,
    )
    parser.add_argument("mode", nargs="?", choices=sorted(MODES))
    namespace, remaining_args = parser.parse_known_args(argv)

    if namespace.mode == "cli":
        from llm_cache.cli.main import main as run_cli

        return run_cli(remaining_args)

    if namespace.mode == "web" or namespace.mode is None:
        from llm_cache.web.main import main as run_web

        return run_web(remaining_args)

    parser.error("mode must be 'web' or 'cli'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
