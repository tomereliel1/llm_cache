from __future__ import annotations

import argparse
import html
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from llm_cache.config.runtime_config import apply_config_defaults
from llm_cache.orchestrator import OrchestratorGrpcClient, QueryResult

MAX_REQUEST_BYTES = 64 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the web client for the orchestrator gRPC server."
    )
    parser.add_argument("--target", default="localhost:50050")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1", help="Web server bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Web server port.")
    apply_config_defaults(parser, argv, "client")
    args = parser.parse_args(argv)

    args.target = args.target.strip()
    args.host = args.host.strip()
    if not args.target:
        parser.error("--target must not be empty")
    if not args.host:
        parser.error("--host must not be empty")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def render_page(
    *,
    prompt: str = "",
    result: QueryResult | None = None,
    error: str | None = None,
    technical_details: str | None = None,
    ready: bool = True,
) -> bytes:
    safe_prompt = html.escape(prompt, quote=True)
    outcome = ""
    if result is not None:
        cache_label = "Cache hit" if result.cache_hit else "Fresh response"
        cache_class = "hit" if result.cache_hit else "miss"
        outcome = f"""
        <section class="answer" aria-live="polite">
          <div class="answer-head">
            <h2>Answer</h2>
            <span class="badge {cache_class}">{cache_label}</span>
          </div>
          <div class="response">{html.escape(result.response)}</div>
        </section>"""
    elif error:
        details = ""
        if technical_details:
            details = f"""
            <details>
              <summary>View technical details</summary>
              <pre>{html.escape(technical_details)}</pre>
            </details>"""
        outcome = (
            f'<div class="error" role="alert">{html.escape(error)}{details}</div>'
        )

    status_class = "ready" if ready else "waiting"
    status_text = "Backend ready" if ready else "Waiting for backend"
    disabled = "" if ready else " disabled"

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Cache</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; color: #eaf0ff; background:
      radial-gradient(circle at 15% 0%, #24315f 0, transparent 38rem), #0b1020; }}
    main {{ width: min(820px, calc(100% - 32px)); margin: 0 auto; padding: 72px 0; }}
    .eyebrow {{ color: #8ba4ff; font-size: .78rem; font-weight: 750; letter-spacing: .14em;
      text-transform: uppercase; }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(2.3rem, 7vw, 4.2rem); letter-spacing: -.055em; }}
    .intro {{ margin: 0 0 32px; color: #aeb9d3; font-size: 1.08rem; }}
    .topline {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .status {{ display: inline-flex; align-items: center; gap: 8px; border-radius: 999px;
      padding: 7px 11px; color: #b8c3dc; background: #151f38; font-size: .78rem;
      font-weight: 750; white-space: nowrap; }}
    .status::before {{ content: ''; width: 8px; height: 8px; border-radius: 50%; }}
    .status.ready::before {{ background: #4fe0a3; box-shadow: 0 0 10px #4fe0a3; }}
    .status.waiting::before {{ background: #ffbd66; box-shadow: 0 0 10px #ffbd66; }}
    .panel, .answer {{ border: 1px solid #2c3757; border-radius: 18px; background: #121a30dd;
      box-shadow: 0 24px 70px #0005; }}
    .panel {{ padding: 22px; }}
    label {{ display: block; margin-bottom: 10px; font-weight: 700; }}
    textarea {{ width: 100%; min-height: 150px; resize: vertical; padding: 16px; color: #f5f7ff;
      background: #0b1122; border: 1px solid #384568; border-radius: 12px; font: inherit;
      line-height: 1.55; outline: none; }}
    textarea:focus {{ border-color: #7793ff; box-shadow: 0 0 0 3px #7793ff25; }}
    .actions {{ display: flex; justify-content: flex-end; margin-top: 14px; }}
    button {{ border: 0; border-radius: 11px; padding: 12px 20px; color: #081020;
      background: #9eb2ff; font: inherit; font-weight: 800; cursor: pointer; }}
    button:hover {{ background: #b9c6ff; }}
    button:disabled {{ cursor: wait; opacity: .7; }}
    .answer {{ margin-top: 22px; padding: 22px; }}
    .answer-head {{ display: flex; align-items: center; justify-content: space-between;
      gap: 12px; }}
    h2 {{ margin: 0; font-size: 1.1rem; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: .76rem; font-weight: 800; }}
    .hit {{ color: #89edc1; background: #163d34; }} .miss {{ color: #ffd591; background: #49351b; }}
    .response {{ margin-top: 18px; color: #dbe3f7; line-height: 1.68; white-space: pre-wrap; }}
    .error {{ margin-top: 18px; padding: 14px 16px; color: #ffc0c6; background: #401d28;
      border: 1px solid #713341; border-radius: 12px; }}
    details {{ margin-top: 12px; color: #e1a9af; }}
    summary {{ width: fit-content; cursor: pointer; font-weight: 700; }}
    pre {{ overflow-x: auto; margin: 10px 0 0; padding: 12px; color: #f2d9dc;
      background: #28121a; border-radius: 8px; white-space: pre-wrap; word-break: break-word; }}
    footer {{ margin-top: 22px; color: #71809f; font-size: .82rem; text-align: center; }}
  </style>
</head>
<body>
  <main>
    <div class="topline">
      <div class="eyebrow">Semantic cache client</div>
      <div id="backend-status" class="status {status_class}">{status_text}</div>
    </div>
    <h1>Ask once. Reuse wisely.</h1>
    <p class="intro">Send a prompt through your existing distributed cache setup.</p>
    <form class="panel" method="post" action="/">
      <label for="prompt">Your prompt</label>
      <textarea id="prompt" name="prompt" maxlength="10000" required autofocus
        placeholder="What would you like to know?">{safe_prompt}</textarea>
      <div class="actions"><button type="submit"{disabled}>Send prompt</button></div>
    </form>
    {outcome}
    <footer>Connected through the orchestrator gRPC service</footer>
  </main>
  <script>
    const form = document.querySelector('form');
    const button = document.querySelector('button');
    const status = document.querySelector('#backend-status');
    form.addEventListener('submit', () => {{
      button.disabled = true; button.textContent = 'Thinking…';
    }});
    async function checkHealth() {{
      try {{
        const response = await fetch('/health', {{ cache: 'no-store' }});
        const health = await response.json();
        const ready = health.status === 'ready';
        status.className = `status ${{ready ? 'ready' : 'waiting'}}`;
        status.textContent = ready ? 'Backend ready' : 'Waiting for backend';
        if (button.textContent !== 'Thinking…') button.disabled = !ready;
      }} catch (_) {{
        status.className = 'status waiting';
        status.textContent = 'Backend unavailable';
        if (button.textContent !== 'Thinking…') button.disabled = true;
      }}
    }}
    checkHealth();
    setInterval(checkHealth, 3000);
  </script>
</body>
</html>"""
    return document.encode("utf-8")


def make_handler(
    query: Callable[[str], QueryResult],
    is_ready: Callable[[], bool] = lambda: True,
) -> type[BaseHTTPRequestHandler]:
    class WebClientHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                ready = is_ready()
                self._send_json(
                    {"status": "ready" if ready else "unavailable"},
                    status=200 if ready else 503,
                )
                return
            if self.path == "/":
                ready = is_ready()
                self._send_page(render_page(ready=ready))
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if length > MAX_REQUEST_BYTES:
                self.send_error(413, "Request is too large")
                return

            form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            prompt = form.get("prompt", [""])[0].strip()
            if not prompt:
                self._send_page(render_page(error="Prompt must not be empty."), status=400)
                return
            if not is_ready():
                self._send_page(
                    render_page(
                        prompt=prompt,
                        error="The backend is not ready yet. Please try again shortly.",
                        ready=False,
                    ),
                    status=503,
                )
                return
            try:
                result = query(prompt)
                page = render_page(prompt=prompt, result=result)
                self._send_page(page)
            except RuntimeError as error:
                self._send_page(
                    render_page(
                        prompt=prompt,
                        error=str(error),
                        technical_details=getattr(error, "technical_details", None),
                    ),
                    status=502,
                )

        def _send_json(self, payload: dict[str, str], status: int) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_page(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return WebClientHandler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with OrchestratorGrpcClient(args.target, args.timeout_seconds) as client:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            make_handler(client.query, lambda: client.is_ready(timeout_seconds=0.5)),
        )
        print(f"Web client: http://{args.host}:{args.port}")
        print(f"Orchestrator: {args.target}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping web client...")
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
