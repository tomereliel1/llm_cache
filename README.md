# LLM Semantic Cache

This project implements a semantic cache for LLM responses. A client sends a prompt to an
orchestrator, the orchestrator embeds the prompt, searches a vector-store cache, returns a
cache hit when similarity is high enough, and otherwise calls an LLM service and stores the
new prompt/embedding/response tuple.

## Architecture

The main distributed processes are:

- Web or CLI client
- Orchestrator gRPC service
- Embedding gRPC service
- Vector-store gRPC service
- LLM gRPC service

The same orchestrator logic works with local providers and gRPC client adapters through
provider-independent interfaces. Current concrete providers include Ollama embedding, Ollama
LLM, Groq LLM, in-memory vector store, Chroma vector store, LRU eviction, and deterministic
test doubles for automated validation.

## Quick Validation

Install dependencies and run the deterministic submission checks:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
docker compose config
docker compose build
```

The test suite uses isolated ports and test doubles for deterministic gRPC and orchestration
coverage. Live Ollama or Groq runs are optional smoke tests because they require local models
or credentials.

## Local Demo

Deterministic local smoke test without external providers:

```bash
uv run python demos/local_one_prompt_demo.py \
  --embedding-provider embedder-stub \
  --llm-provider llm-provider-spy \
  --vector-store-provider in-memory \
  --prompt "What is semantic caching?"
```

Live Ollama demo:

```bash
ollama serve
ollama pull embeddinggemma
ollama pull gemma3:4b
uv run python demos/local_one_prompt_demo.py
```

## gRPC And Docker

For the containerized distributed workflow, see [docs/DOCKER.md](docs/DOCKER.md).

For vector-store distance functions and threshold behavior, see
[docs/VECTOR_STORE_DISTANCE.md](docs/VECTOR_STORE_DISTANCE.md).

For inspecting cache hits, misses, and per-request traces, see
[docs/LOGGING.md](docs/LOGGING.md).

For evaluating cache behavior on downloaded prompt traces, see
[docs/TRACE_ANALYSIS.md](docs/TRACE_ANALYSIS.md).

## Configuration

Runtime JSON configuration supports separate sections for the embedding service,
vector-store service, LLM service, orchestrator service, web client, and CLI client.
Explicit CLI arguments override JSON defaults. Use:

```bash
uv run python demos/local_one_prompt_demo.py --list-supported-configs
uv run python main.py --help
```

Local machine config belongs in `configs/configuration.json`, copied from one of the
committed examples. It is ignored by Git.

## Known Limitations

The in-memory cache is process-local and non-persistent. Chroma persistence is configurable.
Concurrent in-memory reads and writes are protected against state corruption, but identical
in-flight misses are not coalesced, so two simultaneous first-time prompts can both call the
LLM. Live demos require Ollama models or Groq credentials.
