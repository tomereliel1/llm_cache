from __future__ import annotations

from llm_cache.config import VectorStoreConfig, VectorStoreServerConfig
from llm_cache.vector_store import InMemoryVectorStore, vector_store_grpc_server


class ServerSpy:
    def __init__(self) -> None:
        self.bound_address: str | None = None
        self.started = False
        self.waited = False

    def add_insecure_port(self, bind_address: str) -> int:
        self.bound_address = bind_address
        return 60000

    def start(self) -> None:
        self.started = True

    def wait_for_termination(self) -> None:
        self.waited = True


def test_main_uses_parsed_config_to_start_server(monkeypatch) -> None:
    vector_store = InMemoryVectorStore()
    server = ServerSpy()
    config = VectorStoreServerConfig(
        host="127.0.0.1",
        port=60000,
        max_workers=3,
        vector_store=VectorStoreConfig(provider="in-memory"),
    )
    factory_calls = []
    server_calls = []

    monkeypatch.setattr(
        vector_store_grpc_server,
        "parse_vector_store_server_args",
        lambda argv: config,
    )

    def create_vector_store(vector_store_config: VectorStoreConfig) -> InMemoryVectorStore:
        factory_calls.append(vector_store_config)
        return vector_store

    def create_vector_store_grpc_server(
        vector_store_arg: InMemoryVectorStore,
        max_workers: int,
    ) -> ServerSpy:
        server_calls.append((vector_store_arg, max_workers))
        return server

    monkeypatch.setattr(vector_store_grpc_server, "create_vector_store", create_vector_store)
    monkeypatch.setattr(
        vector_store_grpc_server,
        "create_vector_store_grpc_server",
        create_vector_store_grpc_server,
    )

    assert vector_store_grpc_server.main(["ignored"]) == 0

    assert factory_calls == [config.vector_store]
    assert server_calls == [(vector_store, 3)]
    assert server.bound_address == "127.0.0.1:60000"
    assert server.started is True
    assert server.waited is True
