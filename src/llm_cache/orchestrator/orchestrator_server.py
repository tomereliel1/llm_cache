from __future__ import annotations

import argparse
from concurrent import futures
from contextlib import ExitStack

import grpc

from llm_cache.embedding.embedding_grpc_client import EmbeddingGrpcClient
from llm_cache.llm.llm_grpc_client import LLMGrpcClient
from llm_cache.orchestrator import CacheOrchestrator, orchestrator_pb2_grpc
from llm_cache.orchestrator.orchestrator_grpc_service import OrchestratorGrpcService
from llm_cache.server_output import print_server_started
from llm_cache.vector_store.vector_store_grpc_client import VectorStoreGrpcClient

DEFAULT_EMBEDDING_TARGET = "localhost:50051"
DEFAULT_VECTOR_STORE_TARGET = "localhost:50052"
DEFAULT_LLM_TARGET = "localhost:50053"


def create_orchestrator_server(
    orchestrator: CacheOrchestrator,
    max_workers: int = 10,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcService(orchestrator), server
    )
    return server


def parse_orchestrator_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public orchestrator gRPC server using remote embedding, "
            "vector-store, and LLM gRPC providers."
        ),
        epilog=(
            "Example:\n"
            "  uv run python -m llm_cache.orchestrator.orchestrator_server "
            "--embedding-target localhost:50051 "
            "--vector-store-target localhost:50052 --llm-target localhost:50053"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--embedding-target", default=DEFAULT_EMBEDDING_TARGET)
    parser.add_argument("--vector-store-target", default=DEFAULT_VECTOR_STORE_TARGET)
    parser.add_argument("--llm-target", default=DEFAULT_LLM_TARGET)
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for calls from the orchestrator to provider services. Default: 300",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Check that all three provider gRPC targets are reachable, then exit.",
    )
    args = parser.parse_args(argv)

    args.host = args.host.strip()
    for name in ("embedding_target", "vector_store_target", "llm_target"):
        value = getattr(args, name).strip()
        if not value:
            parser.error(f"--{name.replace('_', '-')} must not be empty")
        setattr(args, name, value)
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.provider_timeout_seconds <= 0:
        parser.error("--provider-timeout-seconds must be positive")
    return args


def check_provider_targets(
    providers: list[tuple[str, str]],
    timeout_seconds: float,
) -> bool:
    all_ready = True
    for provider_type, target in providers:
        channel = grpc.insecure_channel(target)
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout_seconds)
            print(f"[OK] {provider_type} provider reachable: {target}")
        except grpc.FutureTimeoutError:
            print(f"[FAILED] {provider_type} provider unreachable: {target}")
            all_ready = False
        finally:
            channel.close()
    return all_ready


def main(argv: list[str] | None = None) -> int:
    args = parse_orchestrator_server_args(argv)
    providers = [
        ("Embedding", args.embedding_target),
        ("Vector-store", args.vector_store_target),
        ("LLM", args.llm_target),
    ]
    if args.check_setup:
        return 0 if check_provider_targets(providers, args.provider_timeout_seconds) else 1

    with ExitStack() as stack:
        embedder = stack.enter_context(
            EmbeddingGrpcClient(args.embedding_target, args.provider_timeout_seconds)
        )
        vector_store = stack.enter_context(
            VectorStoreGrpcClient(args.vector_store_target, args.provider_timeout_seconds)
        )
        llm_provider = stack.enter_context(
            LLMGrpcClient(args.llm_target, args.provider_timeout_seconds)
        )
        orchestrator = CacheOrchestrator(embedder, llm_provider, vector_store)
        server = create_orchestrator_server(orchestrator, args.max_workers)
        address = f"{args.host}:{args.port}"
        if server.add_insecure_port(address) == 0:
            print(f"Could not bind orchestrator gRPC server to {address}")
            return 1
        server.start()
        print_server_started(
            "Orchestrator",
            address,
            (
                ("Embedding target", args.embedding_target),
                ("Vector Store target", args.vector_store_target),
                ("LLM target", args.llm_target),
            ),
        )
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("Stopping orchestrator gRPC server...")
            server.stop(grace=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
