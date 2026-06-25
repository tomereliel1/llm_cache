from __future__ import annotations

import argparse

from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import EmbedderStub, VectorStoreMissStub


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: CacheOrchestrator with a remote LLM gRPC service."
    )
    parser.add_argument("--llm-target", default="localhost:50053")
    parser.add_argument("--prompt", default="What is a semantic cache?")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with LLMGrpcClient(target=args.llm_target) as llm_provider:
        orchestrator = CacheOrchestrator(
            embedder=EmbedderStub(),
            llm_provider=llm_provider,
            vector_store=VectorStoreMissStub(),
        )
        result = orchestrator.query(args.prompt)

    print(f"Prompt: {args.prompt}")
    print(f"Answer: {result.response}")
    print(f"Cache hit: {result.cache_hit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
