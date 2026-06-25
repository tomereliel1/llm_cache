from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.test_doubles import EmbedderStub, LLMProviderSpy
from llm_cache.vector_store import InMemoryVectorStore, VectorStoreGrpcClient
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: CacheOrchestrator with remote vector store gRPC service."
    )
    parser.add_argument("--vector-store-target", default=None)
    parser.add_argument("--prompt", default="What is a semantic cache?")
    return parser.parse_args(argv)


@contextmanager
def local_vector_store_target() -> Iterator[str]:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    server = create_vector_store_grpc_server(vector_store, max_workers=2)
    port = server.add_insecure_port("localhost:0")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        server.stop(grace=0)


def run_demo(prompt: str, vector_store_target: str) -> tuple[bool, bool, int]:
    embedder = EmbedderStub(vector=[0.1, 0.2, 0.3])
    llm_provider = LLMProviderSpy(answer="demo answer from local LLM test double")

    with VectorStoreGrpcClient(target=vector_store_target) as vector_store:
        orchestrator = CacheOrchestrator(
            embedder=embedder,
            llm_provider=llm_provider,
            vector_store=vector_store,
        )

        first_result = orchestrator.query(prompt)
        second_result = orchestrator.query(prompt)

    return first_result.cache_hit, second_result.cache_hit, llm_provider.calls_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.vector_store_target is not None:
        first_hit, second_hit, llm_calls = run_demo(args.prompt, args.vector_store_target)
    else:
        with local_vector_store_target() as target:
            first_hit, second_hit, llm_calls = run_demo(args.prompt, target)

    print(f"Prompt: {args.prompt}")
    print(f"First query cache hit: {first_hit}")
    print(f"Second query cache hit: {second_hit}")
    print(f"LLM calls: {llm_calls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
