from __future__ import annotations

import argparse

from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import LLMProviderSpy, VectorStoreMissStub


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: CacheOrchestrator with remote embedding gRPC service."
    )
    parser.add_argument("--embedding-target", default="localhost:50051")
    parser.add_argument("--prompt", default="What is a semantic cache?")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with EmbeddingGrpcClient(target=args.embedding_target) as embedder:
        vector_store = VectorStoreMissStub()
        llm_provider = LLMProviderSpy(answer="demo answer from local LLM test double")
        orchestrator = CacheOrchestrator(
            embedder=embedder,
            llm_provider=llm_provider,
            vector_store=vector_store,
        )

        result = orchestrator.query(args.prompt)

    print(f"Prompt: {args.prompt}")
    print(f"Answer: {result.response}")
    print(f"Cache hit: {result.cache_hit}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
