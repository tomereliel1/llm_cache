from __future__ import annotations

import argparse
from collections.abc import Callable

from llm_cache.orchestrator import OrchestratorGrpcClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send prompts to an orchestrator gRPC server.")
    parser.add_argument("--target", default="localhost:50050")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not args.target.strip():
        parser.error("--target must not be empty")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def run_prompt_loop(
    client: OrchestratorGrpcClient,
    read_input: Callable[[str], str] = input,
) -> None:
    while True:
        try:
            prompt = read_input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if prompt.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not prompt.strip():
            print("Prompt must not be empty.")
            continue
        try:
            result = client.query(prompt)
        except RuntimeError as error:
            print(error)
            continue
        print("Response:")
        print(result.response)
        print(f"Cache hit: {str(result.cache_hit).lower()}")
        if result.score is not None:
            print(f"Similarity score: {result.score}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Connecting to orchestrator at {args.target}")
    print("Type a prompt, or 'exit'/'quit' to quit.\n")
    with OrchestratorGrpcClient(args.target, args.timeout_seconds) as client:
        run_prompt_loop(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
