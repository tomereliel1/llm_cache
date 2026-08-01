import json

from llm_cache.orchestrator import QueryResult
from llm_cache.web.main import parse_args, render_page


def test_web_args_keep_existing_client_target_and_timeout() -> None:
    args = parse_args(
        [
            "--target",
            "server:123",
            "--timeout-seconds",
            "12",
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
        ]
    )

    assert args.target == "server:123"
    assert args.timeout_seconds == 12
    assert args.host == "0.0.0.0"
    assert args.port == 9090


def test_web_args_load_from_web_client_config_section(tmp_path) -> None:
    config_path = tmp_path / "configuration.json"
    config_path.write_text(
        json.dumps(
            {
                "web_client": {
                    "target": "orchestrator:50050",
                    "timeout_seconds": 300.0,
                    "host": "0.0.0.0",
                    "port": 8080,
                }
            }
        )
    )

    args = parse_args(["--config", str(config_path)])

    assert args.target == "orchestrator:50050"
    assert args.host == "0.0.0.0"
    assert args.port == 8080


def test_render_page_shows_response_and_cache_status() -> None:
    page = render_page(
        prompt="same prompt",
        result=QueryResult(response="cached answer", cache_hit=True),
    ).decode()

    assert "cached answer" in page
    assert "Cache hit" in page
    assert "same prompt" in page
    assert 'id="latest-result"' in page


def test_render_page_includes_session_history_shell() -> None:
    page = render_page().decode()

    assert 'id="history"' in page
    assert "Recent prompts" in page
    assert "sessionStorage" in page
    assert "items.slice(0, 10)" in page


def test_render_page_serializes_successful_result_for_session_history() -> None:
    page = render_page(
        prompt="remember me",
        result=QueryResult(response="answer me", cache_hit=False),
    ).decode()

    assert '"prompt": "remember me"' in page
    assert '"response": "answer me"' in page
    assert '"cacheHit": false' in page


def test_render_page_expands_selected_history_item_in_place() -> None:
    page = render_page().decode()

    assert "entry.setAttribute('aria-expanded', 'false');" in page
    assert "historyList.querySelectorAll('.history-item')" in page
    assert "entry.setAttribute('aria-expanded', String(!isExpanded));" in page
    assert "promptInput.focus();" not in page


def test_render_page_does_not_save_errors_to_session_history() -> None:
    page = render_page(
        prompt="failed prompt",
        error="LLM service is unavailable.",
    ).decode()

    assert "const latestResult = null;" in page


def test_render_page_escapes_user_and_model_content() -> None:
    page = render_page(
        prompt='<script>alert("prompt")</script>',
        result=QueryResult(response="<img src=x onerror=alert(1)>", cache_hit=False),
    ).decode()

    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;img src=x" in page


def test_render_page_json_encodes_history_content_without_html_script_breakout() -> None:
    page = render_page(
        prompt='</script><script>alert("prompt")</script>',
        result=QueryResult(
            response='</script><img src=x onerror=alert("response")>',
            cache_hit=True,
        ),
    ).decode()

    assert "</script><script>alert" not in page
    assert "</script><img" not in page
    assert "\\u003c/script\\u003e" in page


def test_render_page_disables_submission_while_backend_is_unavailable() -> None:
    page = render_page(ready=False).decode()

    assert "Waiting for orchestrator" in page
    assert 'type="submit" disabled' in page


def test_render_page_hides_escaped_technical_error_details_by_default() -> None:
    page = render_page(
        error="LLM service is unavailable.",
        technical_details="DNS lookup failed for <llm-service>",
    ).decode()

    assert "LLM service is unavailable." in page
    assert "View technical details" in page
    assert "<details>" in page
    assert "DNS lookup failed for &lt;llm-service&gt;" in page
    assert "DNS lookup failed for <llm-service>" not in page
