# Run the fully distributed LLM cache

The root `main.py` is the interactive client. It calls the orchestrator through gRPC,
and the orchestrator calls the embedding, vector-store, and LLM services through gRPC.
The original local one-process application remains available as
`demos.local_one_prompt_demo`.

## Prerequisites

1. From the repository root, run `uv sync`.
2. Install Ollama and ensure the `ollama` command is available.
3. Pull the models:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

## Start the system

Use 6 terminals, or 5 if Ollama is already running.

### Terminal 1 — Ollama

```bash
ollama serve
```

### Terminal 2 — embedding gRPC server

```bash
uv run python -m llm_cache.embedding.grpc.server \
  --host localhost --port 50051 \
  --provider ollama --model embeddinggemma
```

### Terminal 3 — vector-store gRPC server

For a cache that lasts until this process exits:

```bash
uv run python -m llm_cache.vector_store.grpc.server \
  --host localhost --port 50052 \
  --provider in-memory \
  --similarity-threshold 0.8 --capacity 1000
```

Use `--provider chroma` for persistent storage. Its default location is
`.cache/vector_store`.

### Terminal 4 — LLM gRPC server

```bash
uv run python -m llm_cache.llm.grpc.server \
  --host localhost --port 50053 \
  --provider ollama --model gemma3:4b
```

### Terminal 5 — orchestrator gRPC server

```bash
uv run python -m llm_cache.orchestrator.grpc.server \
  --host localhost --port 50050 \
  --embedding-target localhost:50051 \
  --vector-store-target localhost:50052 \
  --llm-target localhost:50053
```

The three targets may use different hosts or ports. To check that all provider targets
are reachable without starting the orchestrator server, add `--check-setup`.

### Terminal 6 — interactive user client

```bash
uv run python main.py --target localhost:50050
```

Equivalent module command:

```bash
uv run python -m llm_cache.cli.main --target localhost:50050
```

Enter any number of prompts. Enter `exit` or `quit`, press Ctrl+D, or press Ctrl+C to
close the client. Stop each server with Ctrl+C.

## Request path

```text
main.py
  -> orchestrator gRPC server (:50050)
      -> embedding gRPC server (:50051)
      -> vector-store gRPC server (:50052)
      -> LLM gRPC server (:50053), only on a cache miss
```
