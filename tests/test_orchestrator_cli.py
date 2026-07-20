import json
from argparse import Namespace

from llm_cache.cli.main import parse_args, run_prompt_loop
from llm_cache.orchestrator import QueryResult
from llm_cache.orchestrator.grpc import server as orchestrator_server
from llm_cache.orchestrator.grpc.server import (
    check_provider_targets,
    parse_orchestrator_server_args,
)


class FakeClient:
    def __init__(self):
        self.prompts = []

    def query(self, prompt):
        self.prompts.append(prompt)
        return QueryResult("answer", False)


class ServerSpy:
    def __init__(self):
        self.bound_address = None
        self.started = False
        self.waited = False

    def add_insecure_port(self, bind_address):
        self.bound_address = bind_address
        return 1

    def start(self):
        self.started = True

    def wait_for_termination(self):
        self.waited = True


class ProviderClientSpy:
    def __init__(self, target, timeout_seconds):
        self.target = target
        self.timeout_seconds = timeout_seconds
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True


def make_server_args(check_setup=False):
    return Namespace(
        host="127.0.0.1",
        port=60050,
        workers=4,
        embedding_target="embedding:50051",
        vector_store_target="vector:50052",
        llm_target="llm:50053",
        provider_timeout_seconds=12,
        check_setup=check_setup,
    )


def test_client_args_parse_target_and_timeout():
    args = parse_args(["--target", "server:123", "--timeout-seconds", "2"])
    assert args.target == "server:123"
    assert args.timeout_seconds == 2


def test_client_args_load_from_cli_client_config_section(tmp_path):
    config_path = tmp_path / "configuration.json"
    config_path.write_text(
        json.dumps(
            {
                "cli_client": {
                    "target": "orchestrator:50050",
                    "timeout_seconds": 300.0,
                }
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.target == "orchestrator:50050"
    assert args.timeout_seconds == 300.0


def test_prompt_loop_ignores_blank_and_exits(capsys):
    values = iter(["", "hello", "quit"])
    client = FakeClient()
    run_prompt_loop(client, lambda _: next(values))

    assert client.prompts == ["hello"]
    assert "Prompt must not be empty." in capsys.readouterr().out


def test_server_args_include_runtime_and_provider_targets():
    args = parse_orchestrator_server_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "50100",
            "--workers",
            "4",
            "--embedding-target",
            "embedding:51051",
            "--vector-store-target",
            "vectors:51052",
            "--llm-target",
            "llm:51053",
            "--provider-timeout-seconds",
            "12",
        ]
    )
    assert (args.host, args.port, args.workers) == ("127.0.0.1", 50100, 4)
    assert args.embedding_target == "embedding:51051"
    assert args.vector_store_target == "vectors:51052"
    assert args.llm_target == "llm:51053"
    assert args.provider_timeout_seconds == 12


def test_provider_target_check_prints_provider_name(monkeypatch, capsys):
    class FakeChannel:
        def close(self):
            pass

    class ReadyFuture:
        def result(self, timeout):
            assert timeout == 5

    monkeypatch.setattr(orchestrator_server.grpc, "insecure_channel", lambda target: FakeChannel())
    monkeypatch.setattr(
        orchestrator_server.grpc,
        "channel_ready_future",
        lambda channel: ReadyFuture(),
    )

    assert check_provider_targets([("LLM", "llm:50053")], timeout_seconds=5) is True
    assert capsys.readouterr().out == "[OK] LLM provider reachable: llm:50053\n"


def test_orchestrator_setup_check_failure_prevents_server_start(monkeypatch):
    args = make_server_args(check_setup=True)
    server_created = False

    monkeypatch.setattr(orchestrator_server, "parse_orchestrator_server_args", lambda argv: args)
    monkeypatch.setattr(
        orchestrator_server,
        "check_provider_targets",
        lambda providers, timeout_seconds: False,
    )

    def create_orchestrator_server(orchestrator, max_workers):
        nonlocal server_created
        server_created = True

    monkeypatch.setattr(
        orchestrator_server,
        "create_orchestrator_server",
        create_orchestrator_server,
    )

    assert orchestrator_server.main([]) == 1
    assert server_created is False


def test_orchestrator_setup_check_success_starts_server(monkeypatch):
    args = make_server_args(check_setup=True)
    server = ServerSpy()
    checked = []
    clients = []

    monkeypatch.setattr(orchestrator_server, "parse_orchestrator_server_args", lambda argv: args)

    def check_provider_targets(providers, timeout_seconds):
        checked.append((providers, timeout_seconds))
        return True

    def create_client(target, timeout_seconds):
        client = ProviderClientSpy(target, timeout_seconds)
        clients.append(client)
        return client

    monkeypatch.setattr(orchestrator_server, "check_provider_targets", check_provider_targets)
    monkeypatch.setattr(orchestrator_server, "EmbeddingGrpcClient", create_client)
    monkeypatch.setattr(orchestrator_server, "VectorStoreGrpcClient", create_client)
    monkeypatch.setattr(orchestrator_server, "LLMGrpcClient", create_client)
    monkeypatch.setattr(
        orchestrator_server,
        "create_orchestrator_server",
        lambda orchestrator, max_workers: server,
    )

    assert orchestrator_server.main([]) == 0
    assert checked == [
        (
            [
                ("Embedding", "embedding:50051"),
                ("Vector-store", "vector:50052"),
                ("LLM", "llm:50053"),
            ],
            12,
        )
    ]
    assert server.bound_address == "127.0.0.1:60050"
    assert server.started is True
    assert server.waited is True
    assert all(client.entered and client.exited for client in clients)
