from __future__ import annotations

import sys

from llm_cache.config import ConfigError
from llm_cache.config.vector_store_server_cli_args import parse_vector_store_server_args
from llm_cache.factories.vector_store_factory import create_vector_store
from llm_cache.vector_store.vector_store_grpc_service import create_vector_store_grpc_server


def main(argv: list[str] | None = None) -> int:
    config = parse_vector_store_server_args(argv)
    try:
        vector_store = create_vector_store(config.vector_store)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    server = create_vector_store_grpc_server(vector_store, max_workers=config.max_workers)
    address = f"{config.host}:{config.port}"
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind vector store gRPC server to {address}")

    server.start()
    print(f"Vector store gRPC server listening on {address}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
