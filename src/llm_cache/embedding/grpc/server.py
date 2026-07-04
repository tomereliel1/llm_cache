from __future__ import annotations

from concurrent import futures
from typing import Any

import grpc

from llm_cache.config.embedding_server_cli_args import parse_embedding_server_args
from llm_cache.embedding.grpc.generated import embedding_pb2_grpc
from llm_cache.embedding.grpc.service import EmbeddingGrpcService
from llm_cache.factories import create_embedder
from llm_cache.health import all_healthy, format_health_report, run_health_checks
from llm_cache.server_output import print_server_started


def create_embedding_server(
    embedder: Any,
    max_workers: int,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGrpcService(embedder),
        server,
    )
    return server


def main(argv: list[str] | None = None) -> int:
    config = parse_embedding_server_args(argv)

    embedder = create_embedder(config.embedding)

    if config.check_setup:
        results = run_health_checks([embedder])
        print(format_health_report(results))
        if not all_healthy(results):
            return 1

    server = create_embedding_server(embedder, max_workers=config.max_workers)
    bind_address = f"{config.host}:{config.port}"
    server.add_insecure_port(bind_address)
    server.start()

    print_server_started(
        "Embedding",
        bind_address,
        (
            ("Provider", config.embedding.provider),
            ("Model", config.embedding.model),
        ),
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Stopping embedding gRPC server...")
        server.stop(grace=1)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
