import subprocess
import sys


def test_importing_main_does_not_import_provider_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import main; print('chromadb' in sys.modules, 'ollama' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"


def test_importing_cli_main_does_not_import_provider_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import llm_cache.cli.main; "
                "print('chromadb' in sys.modules, 'ollama' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"


def test_importing_orchestrator_server_does_not_import_provider_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import llm_cache.orchestrator.grpc.server; "
                "print('chromadb' in sys.modules, 'ollama' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"


def test_orchestrator_with_test_doubles_does_not_import_optional_providers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from llm_cache.orchestrator import CacheOrchestrator; "
                "from llm_cache.test_doubles import "
                "EmbedderStub, LLMProviderSpy, VectorStoreMissStub; "
                "orchestrator = CacheOrchestrator("
                "EmbedderStub(), LLMProviderSpy(), VectorStoreMissStub()); "
                "orchestrator.query('hello'); "
                "print('chromadb' in sys.modules, 'ollama' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"


def test_main_help_describes_web_client() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "web client" in result.stdout
    assert "--target" in result.stdout
    assert "--port" in result.stdout


def test_main_cli_help_describes_terminal_client() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Send prompts to an orchestrator gRPC server" in result.stdout
    assert "--target" in result.stdout
    assert "--port" not in result.stdout
