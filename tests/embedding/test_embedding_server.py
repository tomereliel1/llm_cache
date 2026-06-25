from __future__ import annotations

from llm_cache.config import EmbeddingConfig, EmbeddingServerConfig
from llm_cache.embedding import embedding_server
from llm_cache.health import HealthCheckResult
from llm_cache.test_doubles import EmbedderStub


class ServerSpy:
    def __init__(self) -> None:
        self.bound_address: str | None = None
        self.started = False
        self.waited = False

    def add_insecure_port(self, bind_address: str) -> None:
        self.bound_address = bind_address

    def start(self) -> None:
        self.started = True

    def wait_for_termination(self) -> None:
        self.waited = True


def test_main_uses_parsed_config_to_start_server(monkeypatch) -> None:
    embedder = EmbedderStub()
    server = ServerSpy()
    config = EmbeddingServerConfig(
        host="127.0.0.1",
        port=60000,
        max_workers=3,
        check_setup=False,
        embedding=EmbeddingConfig(provider="ollama", model="embeddinggemma"),
    )
    factory_calls = []
    server_calls = []

    monkeypatch.setattr(
        embedding_server,
        "parse_embedding_server_args",
        lambda argv: config,
    )

    def create_embedder(embedding_config: EmbeddingConfig) -> EmbedderStub:
        factory_calls.append(embedding_config)
        return embedder

    def create_embedding_server(embedder_arg, max_workers: int):
        server_calls.append((embedder_arg, max_workers))
        return server

    monkeypatch.setattr(embedding_server, "create_embedder", create_embedder)
    monkeypatch.setattr(embedding_server, "create_embedding_server", create_embedding_server)

    assert embedding_server.main(["ignored"]) == 0

    assert factory_calls == [config.embedding]
    assert server_calls == [(embedder, 3)]
    assert server.bound_address == "127.0.0.1:60000"
    assert server.started is True
    assert server.waited is True


def test_main_returns_one_when_health_check_fails(monkeypatch) -> None:
    embedder = EmbedderStub()
    config = EmbeddingServerConfig(
        host="127.0.0.1",
        port=60000,
        max_workers=3,
        check_setup=True,
        embedding=EmbeddingConfig(provider="ollama", model="embeddinggemma"),
    )
    server_created = False

    monkeypatch.setattr(
        embedding_server,
        "parse_embedding_server_args",
        lambda argv: config,
    )
    monkeypatch.setattr(embedding_server, "create_embedder", lambda embedding_config: embedder)
    monkeypatch.setattr(
        embedding_server,
        "run_health_checks",
        lambda components: [
            HealthCheckResult.fail(
                name="embedding",
                message="not healthy",
            )
        ],
    )

    def create_embedding_server(embedder_arg, max_workers: int):
        nonlocal server_created
        server_created = True

    monkeypatch.setattr(embedding_server, "create_embedding_server", create_embedding_server)

    assert embedding_server.main([]) == 1
    assert server_created is False


def test_main_starts_server_after_successful_health_check(monkeypatch) -> None:
    embedder = EmbedderStub()
    server = ServerSpy()
    config = EmbeddingServerConfig(
        host="127.0.0.1",
        port=60000,
        max_workers=3,
        check_setup=True,
        embedding=EmbeddingConfig(provider="ollama", model="embeddinggemma"),
    )

    monkeypatch.setattr(
        embedding_server,
        "parse_embedding_server_args",
        lambda argv: config,
    )
    monkeypatch.setattr(embedding_server, "create_embedder", lambda embedding_config: embedder)
    monkeypatch.setattr(
        embedding_server,
        "run_health_checks",
        lambda components: [
            HealthCheckResult.ok(
                name="embedding",
                message="healthy",
            )
        ],
    )
    monkeypatch.setattr(
        embedding_server,
        "create_embedding_server",
        lambda embedder_arg, max_workers: server,
    )

    assert embedding_server.main([]) == 0
    assert server.started is True
