# User Guide

This guide explains how to install, configure, and run the LLM Semantic Cache project.

The project runs a semantic cache in front of an LLM. When you send a prompt, the system
embeds the prompt, searches a vector-store cache, and either returns a cached answer or
calls the LLM and stores the new answer for future similar prompts.

## Requirements

Required for all local development workflows:

- Python 3.12 or newer
- `uv` for Python dependency management
- Git

Required for the recommended Docker workflow:

- Docker
- Docker Compose

Required for the default Ollama provider:

- Ollama installed locally, or the optional Ollama service in Docker Compose
- The `embeddinggemma` embedding model
- The `gemma3:4b` LLM model

Optional:

- A Groq API key if you choose the Groq LLM provider

The Python dependencies are declared in [pyproject.toml](../pyproject.toml). The main runtime
libraries are `grpcio`, `protobuf`, `ollama`, `groq`, and `chromadb`.

## Install The Project

Clone the repository and install dependencies from the repository root:

```bash
git clone https://github.com/tomereliel1/llm_cache.git
cd llm_cache
uv sync
```

Run a quick deterministic demo that does not require Ollama, Groq, or Docker:

```bash
uv run python demos/local_one_prompt_demo.py \
  --embedding-provider embedder-stub \
  --llm-provider llm-provider-spy \
  --vector-store-provider in-memory \
  --prompt "What is semantic caching?"
```

This confirms that the Python environment and core cache flow are working.

## Configuration

Runtime configuration is stored in JSON files under [configs](../configs). The repository
includes examples, while your local editable file should be named
`configs/configuration.json`. That file is ignored by Git so each machine can keep its own
ports, providers, model names, and secrets.

For a normal local multi-terminal run:

```bash
cp configs/configuration_example.json configs/configuration.json
```

For the recommended Docker Compose run:

```bash
cp configs/configuration_docker_example.json configs/configuration.json
```

Important configuration sections:

```json
{
  "embedding_service": {
    "provider": "ollama",
    "model": "embeddinggemma"
  },
  "vector_store_service": {
    "provider": "chroma",
    "similarity_threshold": 0.8,
    "capacity": 1000,
    "persistent": false
  },
  "llm_service": {
    "provider": "ollama",
    "model": "gemma3:4b"
  },
  "orchestrator_service": {
    "embedding_target": "embedding-service:50051",
    "vector_store_target": "vector-store-service:50052",
    "llm_target": "llm-service:50053"
  },
  "web_client": {
    "target": "orchestrator:50050",
    "port": 8080
  }
}
```

Command-line arguments override values loaded from the JSON file. For example:

```bash
uv run python main.py web --config configs/configuration.json --port 8081
```

For more detail about configuration fields and provider defaults, see
[CONFIGURATION_CHANGE_GUIDE.md](CONFIGURATION_CHANGE_GUIDE.md). For vector-store threshold
and distance behavior, see [VECTOR_STORE_DISTANCE.md](VECTOR_STORE_DISTANCE.md).

## Prepare Ollama Models

The default configuration uses Ollama for both embeddings and LLM responses.

Start Ollama:

```bash
ollama serve
```

In another terminal, pull the required models:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

Keep `ollama serve` running while using the project.

## Recommended Run: Docker Compose

Docker Compose starts the web client, orchestrator, embedding service, vector-store service,
and LLM service as one stack.

First, create the Docker configuration:

```bash
cp configs/configuration_docker_example.json configs/configuration.json
```

Make sure Ollama is running on the host and the models are pulled:

```bash
ollama serve
ollama pull embeddinggemma
ollama pull gemma3:4b
```

Start the system:

```bash
docker compose up --build
```

Open the web client:

```text
http://127.0.0.1:8080
```

Try this usage flow:

1. Enter `What is semantic caching?`
2. Submit the prompt.
3. The first result should be marked `Fresh response`.
4. Submit the same prompt again.
5. The second result should be marked `Cache hit`.

Stop the stack with `Ctrl+C`, then optionally clean up containers:

```bash
docker compose down
```

For more Docker details, including running Ollama inside Compose and troubleshooting
container networking, see [DOCKER.md](DOCKER.md).

## Optional: Run The CLI Client In Docker

The Docker stack also includes an optional interactive CLI client.

Start the backend and web stack:

```bash
docker compose up --build
```

In another terminal, start the CLI client:

```bash
docker compose --profile cli run --rm cli-client
```

Then type prompts:

```text
> What is semantic caching?
Response:
...
Cache hit: false

> What is semantic caching?
Response:
...
Cache hit: true
```

Type `exit` or `quit` to leave the CLI.

## Manual Run Without Docker

The manual workflow is useful when you want to see or debug each service separately. It uses
five or six terminals.

Create the local configuration:

```bash
cp configs/configuration_example.json configs/configuration.json
```

Terminal 1: start Ollama.

```bash
ollama serve
```

Terminal 2: start the embedding gRPC service.

```bash
uv run python -m llm_cache.embedding.grpc.server \
  --host localhost --port 50051 \
  --provider ollama --model embeddinggemma
```

Terminal 3: start the vector-store gRPC service.

For an in-memory cache:

```bash
uv run python -m llm_cache.vector_store.grpc.server \
  --host localhost --port 50052 \
  --provider in-memory \
  --similarity-threshold 0.8 \
  --capacity 1000
```

For a Chroma vector store:

```bash
uv run python -m llm_cache.vector_store.grpc.server \
  --host localhost --port 50052 \
  --provider chroma \
  --path data/chroma \
  --collection llm_cache \
  --distance-function l2 \
  --similarity-threshold 0.8 \
  --capacity 1000
```

Terminal 4: start the LLM gRPC service.

```bash
uv run python -m llm_cache.llm.grpc.server \
  --host localhost --port 50053 \
  --provider ollama --model gemma3:4b
```

Terminal 5: start the orchestrator gRPC service.

```bash
uv run python -m llm_cache.orchestrator.grpc.server \
  --host localhost --port 50050 \
  --embedding-target localhost:50051 \
  --vector-store-target localhost:50052 \
  --llm-target localhost:50053
```

Terminal 6: start the web client.

```bash
uv run python main.py web --target localhost:50050 --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

You can also use the interactive terminal client instead of the web client:

```bash
uv run python main.py cli --target localhost:50050
```

For a shorter manual runbook, see [../RUN_ORCHESTRATOR.md](../RUN_ORCHESTRATOR.md).

## Running With A JSON Config File

Every service and client can load its own section from the shared config file.

Example local run using `configs/configuration.json`:

```bash
uv run python -m llm_cache.embedding.grpc.server --config configs/configuration.json
uv run python -m llm_cache.vector_store.grpc.server --config configs/configuration.json
uv run python -m llm_cache.llm.grpc.server --config configs/configuration.json
uv run python -m llm_cache.orchestrator.grpc.server --config configs/configuration.json
uv run python main.py web --config configs/configuration.json
```

Each command reads only the section it needs. For example, the embedding server reads
`embedding_service`, while the web client reads `web_client`.

## Optional: Use Groq For The LLM Service

To use Groq instead of Ollama for answer generation, set an environment variable:

```bash
export GROQ_API_KEY=<your-api-key>
```

Then configure the LLM service:

```json
{
  "llm_service": {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "groq_api_key_env": "GROQ_API_KEY"
  }
}
```

The config stores the environment variable name, not the secret itself. In Docker Compose,
`GROQ_API_KEY` is passed through to the `llm-service` container. You can also copy
[.env.example](../.env.example) to `.env` and fill in the value there.

## Useful Commands

Show supported providers, models, and defaults:

```bash
uv run python demos/local_one_prompt_demo.py --list-supported-configs
```

Show help for the web client:

```bash
uv run python main.py web --help
```

Show help for the CLI client:

```bash
uv run python main.py cli --help
```

Check the Docker Compose configuration:

```bash
docker compose config
```

Run the automated test suite:

```bash
uv run pytest
```


## Troubleshooting

If `ollama` is not found, install Ollama and make sure the command is available in your
terminal.

If a model is missing, pull it:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

If Docker cannot reach host Ollama, make sure `ollama serve` is running and that
`configs/configuration.json` uses this base URL in both Ollama-backed service sections:

```json
{
  "embedding_service": {
    "base_url": "http://host.docker.internal:11434"
  },
  "llm_service": {
    "base_url": "http://host.docker.internal:11434"
  }
}
```

If port `8080` is already in use, either change the web client port:

```bash
uv run python main.py web --port 8081
```

or change the left side of the port mapping in [docker-compose.yml](../docker-compose.yml):

```yaml
ports:
  - "8081:8080"
```

If a repeated prompt is not a cache hit, check the vector-store provider, persistence setting,
distance function, and similarity threshold. More details are in
[VECTOR_STORE_DISTANCE.md](VECTOR_STORE_DISTANCE.md).

If you need to inspect request flow across services, cache hits, misses, and request IDs, see
[LOGGING.md](LOGGING.md).
