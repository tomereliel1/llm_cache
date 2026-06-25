from __future__ import annotations

import argparse

from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.vector_store.vector_store_grpc_client import VectorStoreGrpcClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demo: CacheOrchestrator with remote embedding, LLM, and vector-store services."
        )
    )
    parser.add_argument("--embedding-target", default="localhost:50051")
    parser.add_argument("--vector-store-target", default="localhost:50052")
    parser.add_argument("--llm-target", default="localhost:50053")
    parser.add_argument("--prompt", default="What is a semantic cache?")
    return parser.parse_args(argv)


def run_demo(
    prompt: str,
    embedding_target: str,
    vector_store_target: str,
    llm_target: str,
) -> tuple[bool, bool, str]:
    with (
        EmbeddingGrpcClient(target=embedding_target) as embedder,
        VectorStoreGrpcClient(target=vector_store_target) as vector_store,
        LLMGrpcClient(target=llm_target) as llm_provider,
    ):
        orchestrator = CacheOrchestrator(
            embedder=embedder,
            llm_provider=llm_provider,
            vector_store=vector_store,
        )
        print("Running first query through all three gRPC services...", flush=True)
        first_result = orchestrator.query(prompt)
        print("First query completed. Running the same query to verify the cache...", flush=True)
        second_result = orchestrator.query(prompt)
        print("Second query completed.", flush=True)

    return first_result.cache_hit, second_result.cache_hit, second_result.response


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    first_hit, second_hit, response = run_demo(
        prompt=args.prompt,
        embedding_target=args.embedding_target,
        vector_store_target=args.vector_store_target,
        llm_target=args.llm_target,
    )

    print(f"Prompt: {args.prompt}")
    print(f"Answer: {response}")
    print(f"First query cache hit: {first_hit}")
    print(f"Second query cache hit: {second_hit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
