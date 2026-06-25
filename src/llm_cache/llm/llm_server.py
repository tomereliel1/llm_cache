from __future__ import annotations

from concurrent import futures

import grpc

from llm_cache.config.llm_server_cli_args import parse_llm_server_args
from llm_cache.factories import create_llm_provider
from llm_cache.health import all_healthy, format_health_report, run_health_checks
from llm_cache.llm import ILLMProvider, llm_pb2_grpc
from llm_cache.llm.llm_grpc_service import LLMGrpcService


def create_llm_server(
    llm_provider: ILLMProvider,
    max_workers: int,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMGrpcService(llm_provider),
        server,
    )
    return server


def main(argv: list[str] | None = None) -> int:
    config = parse_llm_server_args(argv)
    llm_provider = create_llm_provider(config.llm)

    if config.check_setup:
        results = run_health_checks([llm_provider])
        print(format_health_report(results))
        if not all_healthy(results):
            return 1

    server = create_llm_server(llm_provider, max_workers=config.max_workers)
    bind_address = f"{config.host}:{config.port}"
    server.add_insecure_port(bind_address)
    server.start()

    print(
        "LLM gRPC server started "
        f"on {bind_address} "
        f"using provider={config.llm.provider}, model={config.llm.model}"
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Stopping LLM gRPC server...")
        server.stop(grace=1)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
