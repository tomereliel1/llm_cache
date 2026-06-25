from __future__ import annotations

from llm_cache.config import LLMConfig, LLMServerConfig
from llm_cache.health import HealthCheckResult
from llm_cache.llm import llm_server
from llm_cache.test_doubles import LLMProviderSpy


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


def make_config(check_setup: bool = False) -> LLMServerConfig:
    return LLMServerConfig(
        host="127.0.0.1",
        port=60001,
        max_workers=4,
        check_setup=check_setup,
        llm=LLMConfig(provider="ollama", model="gemma3:4b"),
    )


def test_main_uses_parsed_config_to_start_server(monkeypatch) -> None:
    config = make_config()
    provider = LLMProviderSpy()
    server = ServerSpy()
    factory_calls = []
    server_calls = []

    monkeypatch.setattr(llm_server, "parse_llm_server_args", lambda argv: config)

    def create_llm_provider(llm_config: LLMConfig) -> LLMProviderSpy:
        factory_calls.append(llm_config)
        return provider

    def create_llm_server(provider_arg, max_workers: int):
        server_calls.append((provider_arg, max_workers))
        return server

    monkeypatch.setattr(llm_server, "create_llm_provider", create_llm_provider)
    monkeypatch.setattr(llm_server, "create_llm_server", create_llm_server)

    assert llm_server.main(["ignored"]) == 0
    assert factory_calls == [config.llm]
    assert server_calls == [(provider, 4)]
    assert server.bound_address == "127.0.0.1:60001"
    assert server.started is True
    assert server.waited is True


def test_main_returns_one_when_health_check_fails(monkeypatch) -> None:
    config = make_config(check_setup=True)
    provider = LLMProviderSpy()
    server_created = False

    monkeypatch.setattr(llm_server, "parse_llm_server_args", lambda argv: config)
    monkeypatch.setattr(llm_server, "create_llm_provider", lambda llm_config: provider)
    monkeypatch.setattr(
        llm_server,
        "run_health_checks",
        lambda components: [HealthCheckResult.fail(name="llm", message="not healthy")],
    )

    def create_llm_server(provider_arg, max_workers: int):
        nonlocal server_created
        server_created = True

    monkeypatch.setattr(llm_server, "create_llm_server", create_llm_server)

    assert llm_server.main([]) == 1
    assert server_created is False


def test_main_starts_server_after_successful_health_check(monkeypatch) -> None:
    config = make_config(check_setup=True)
    provider = LLMProviderSpy()
    server = ServerSpy()

    monkeypatch.setattr(llm_server, "parse_llm_server_args", lambda argv: config)
    monkeypatch.setattr(llm_server, "create_llm_provider", lambda llm_config: provider)
    monkeypatch.setattr(
        llm_server,
        "run_health_checks",
        lambda components: [HealthCheckResult.ok(name="llm", message="healthy")],
    )
    monkeypatch.setattr(
        llm_server,
        "create_llm_server",
        lambda provider_arg, max_workers: server,
    )

    assert llm_server.main([]) == 0
    assert server.started is True
