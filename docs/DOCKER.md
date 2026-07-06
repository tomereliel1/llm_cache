# Running the distributed cache with Docker

Docker Compose starts the web client, embedding, vector-store, LLM, and orchestrator gRPC
services as one stack. The existing local commands and flags still work; Docker is an
additional workflow.

## Recommended: use Ollama on the host

Install Ollama, then prepare its models:

```bash
ollama serve
ollama pull embeddinggemma
ollama pull gemma3:4b
```

From the repository root, create your editable configuration and start the backend:

```bash
cp configs/configuration_docker_example.json configs/configuration.json
docker compose up --build
```

Open `http://127.0.0.1:8080`. Enter a prompt twice: the first response should show
`Fresh response`, and the exact repeated prompt should show `Cache hit`.

The Docker example configuration is committed as a safe template. `configuration.json` is
ignored by Git so each machine can use different addresses and providers without committing
local settings.
Explicit command-line flags override JSON values, so manual development remains available:

```bash
uv run python -m llm_cache.llm.grpc.server --host localhost --port 50053 \
  --provider ollama --model gemma3:4b
uv run python main.py --target localhost:50050
```

When using Groq, keep the secret in the environment. The optional
`llm_service.groq_api_key_env` JSON field (or `--groq-api-key-env`) selects the environment
variable name; it never contains the key itself. If omitted, it defaults to `GROQ_API_KEY`.

Stop the stack while preserving cached Chroma data with `docker compose down`. Delete the cache
volume with `docker compose down -v`.

## Optional: run Ollama in Compose

Change both `base_url` values in `configs/configuration.json` to
`http://ollama:11434`, then run:

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull embeddinggemma
docker compose exec ollama ollama pull gemma3:4b
docker compose --profile ollama up --build
```

## Troubleshooting

- Missing config: run `cp configs/configuration_docker_example.json configs/configuration.json`.
- `connection refused` for `host.docker.internal:11434`: run `ollama serve`, or use the Ollama
  Compose profile.
- Model not found: run the two `ollama pull` commands above (inside `docker compose exec ollama`
  when using the profile).
- Inside Compose, targets use service names such as `embedding-service:50051`; from the host,
  connect to the exposed orchestrator at `localhost:50050`.
- If port 8080 is already in use, change the left side of `8080:8080` in
  `docker-compose.yml`, for example `8081:8080`.
- `docker compose down` preserves Chroma; `docker compose down -v` clears it.

Host-Ollama mode needs two terminals when Ollama is not already running: one for Ollama and
one for the Compose stack. Compose-Ollama mode needs one terminal after its models are pulled.
