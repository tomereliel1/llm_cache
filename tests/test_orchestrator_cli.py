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


def test_client_args_parse_target_and_timeout():
    args = parse_args(["--target", "server:123", "--timeout-seconds", "2"])
    assert args.target == "server:123"
    assert args.timeout_seconds == 2


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
            "--max-workers",
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
    assert (args.host, args.port, args.max_workers) == ("127.0.0.1", 50100, 4)
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
