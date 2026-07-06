from __future__ import annotations

import argparse
import html
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
        outcome = f'<div class="error" role="alert">{html.escape(error)}</div>'

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
    footer {{ margin-top: 22px; color: #71809f; font-size: .82rem; text-align: center; }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Semantic cache client</div>
    <h1>Ask once. Reuse wisely.</h1>
    <p class="intro">Send a prompt through your existing distributed cache setup.</p>
    <form class="panel" method="post" action="/">
      <label for="prompt">Your prompt</label>
      <textarea id="prompt" name="prompt" maxlength="10000" required autofocus
        placeholder="What would you like to know?">{safe_prompt}</textarea>
      <div class="actions"><button type="submit">Send prompt</button></div>
    </form>
    {outcome}
    <footer>Connected through the orchestrator gRPC service</footer>
  </main>
  <script>
    document.querySelector('form').addEventListener('submit', () => {{
      const button = document.querySelector('button');
      button.disabled = true; button.textContent = 'Thinking…';
    }});
  </script>
</body>
</html>"""
    return document.encode("utf-8")


def make_handler(query: Callable[[str], QueryResult]) -> type[BaseHTTPRequestHandler]:
    class WebClientHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            self._send_page(render_page())

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
            try:
                result = query(prompt)
                page = render_page(prompt=prompt, result=result)
                self._send_page(page)
            except RuntimeError as error:
                self._send_page(render_page(prompt=prompt, error=str(error)), status=502)

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
        server = ThreadingHTTPServer((args.host, args.port), make_handler(client.query))
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
