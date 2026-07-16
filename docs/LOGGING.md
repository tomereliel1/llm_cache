# Logging and request tracing

The distributed cache writes structured logs to the console. Docker captures those
logs automatically, so the main way to inspect the system is with `docker compose logs`.

Each user prompt gets a `request_id`. The same `request_id` is passed through gRPC
metadata from the web client to the orchestrator, embedding service, vector-store
service, and LLM service. This lets you trace one prompt across all containers.

## Start the system

From the repository root:

```bash
docker compose up --build
```

Open the web client:

```text
http://127.0.0.1:8080
```

Submit a prompt, then submit the exact same prompt again to verify a cache hit.

## Follow all logs

In another terminal:

```bash
docker compose logs -f
```

Useful variants:

```bash
docker compose logs --tail=100
docker compose logs -f --tail=50
docker compose logs --no-log-prefix
```

`--no-log-prefix` removes the Docker service prefix, which makes request traces easier
to read.

## Follow one service

```bash
docker compose logs -f client
docker compose logs -f orchestrator
docker compose logs -f embedding-service
docker compose logs -f vector-store-service
docker compose logs -f llm-service
```

## Trace one request

Find a request id in the logs:

```text
request_id=c57ef1fd-9ece-4f75-9552-86f8295fe79d
```

Then filter all logs by that id.

PowerShell:

```powershell
docker compose logs --no-log-prefix | Select-String "c57ef1fd-9ece-4f75-9552-86f8295fe79d"
```

Bash:

```bash
docker compose logs --no-log-prefix | grep "c57ef1fd-9ece-4f75-9552-86f8295fe79d"
```

For live filtering in PowerShell:

```powershell
docker compose logs -f --no-log-prefix | Select-String "c57ef1fd-9ece-4f75-9552-86f8295fe79d"
```

## Check cache behavior

Show cache-related events:

PowerShell:

```powershell
docker compose logs --no-log-prefix | Select-String "cache_|vector_search_completed|llm_request"
```

Bash:

```bash
docker compose logs --no-log-prefix | grep -E "cache_|vector_search_completed|llm_request"
```

For the first prompt, you should see:

```text
vector_search_completed found=False
cache_miss
llm_request_received
response_stored
web_prompt_completed cache_hit=False
```

For the repeated prompt, you should see:

```text
vector_search_completed found=True
cache_hit
web_prompt_completed cache_hit=True
```

You should not see `llm_request_received` for the repeated prompt. That means the LLM
was skipped and the response came from the vector-store cache.

## Log format

Log lines use this shape:

```text
timestamp | service_name | level | request_id=<id> | event fields
```

Example:

```text
2026-07-16 11:36:22,407 | vector-store-service | INFO | request_id=c57ef1fd-9ece-4f75-9552-86f8295fe79d | vector_search_completed found=True
```

The important parts are:

- `service_name`: which container produced the log.
- `request_id`: which user prompt this log belongs to.
- event name: what happened, for example `cache_hit`, `cache_miss`, or
  `vector_search_completed`.

## Request path

On a cache miss:

```text
web-client
  -> orchestrator
  -> embedding-service
  -> vector-store-service search
  -> llm-service
  -> vector-store-service store
  -> web-client
```

On a cache hit:

```text
web-client
  -> orchestrator
  -> embedding-service
  -> vector-store-service search
  -> web-client
```

The LLM service is only called on a cache miss.
