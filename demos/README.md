# Demo runbook

Run all commands from the repository root with `uv sync` completed. For demos using
Ollama, install Ollama, run `ollama serve`, and pull the models shown below.

## 1. Local one-prompt cache demo

Purpose: run the original application directly in one process, without gRPC.

Terminals: 2 (or 1 if Ollama is already running).

Terminal 1:

```bash
ollama serve
ollama pull embeddinggemma
ollama pull gemma3:4b
```

Terminal 2:

```bash
uv run python -m demos.local_one_prompt_demo \
  --prompt "What is semantic caching?" \
  --embedding-provider ollama --embedding-model embeddinggemma \
  --llm-provider ollama --llm-model gemma3:4b \
  --vector-store-provider in-memory
```

## 2. Remote embedding provider demo

Purpose: local orchestrator/test doubles with a real embedding gRPC process.

Terminals: 3 (or 2 if Ollama is already running).

```bash
# Terminal 1
ollama serve
ollama pull embeddinggemma

# Terminal 2
uv run python -m llm_cache.embedding.grpc.server --host localhost --port 50051 \
  --provider ollama --model embeddinggemma

# Terminal 3
uv run python -m demos.grpc_embedding_orchestrator_demo \
  --embedding-target localhost:50051
```

## 3. Remote vector-store provider demo

Purpose: verify a miss followed by a hit through the vector-store gRPC adapter.
It is self-contained and starts an in-process server.

Terminals: 1.

```bash
uv run python -m demos.grpc_vector_store_orchestrator_demo
```

## 4. Remote LLM provider demo

Purpose: local orchestrator/test doubles with a real LLM gRPC process.

Terminals: 3 (or 2 if Ollama is already running).

```bash
# Terminal 1
ollama serve
ollama pull gemma3:4b

# Terminal 2
uv run python -m llm_cache.llm.grpc.server --host localhost --port 50053 \
  --provider ollama --model gemma3:4b

# Terminal 3
uv run python -m demos.grpc_llm_orchestrator_demo --llm-target localhost:50053
```

## 5. All remote providers demo

Purpose: run `CacheOrchestrator` locally while embedding, vector storage, and LLM
generation each run behind their own gRPC boundary.

Terminals: 5 (or 4 if Ollama is already running).

```bash
# Terminal 1
ollama serve

# Terminal 2
uv run python -m llm_cache.embedding.grpc.server --port 50051

# Terminal 3
uv run python -m llm_cache.vector_store.grpc.server --port 50052 \
  --provider in-memory

# Terminal 4
uv run python -m llm_cache.llm.grpc.server --port 50053

# Terminal 5
uv run python -m demos.grpc_providers_orchestrator_demo \
  --embedding-target localhost:50051 --vector-store-target localhost:50052 \
  --llm-target localhost:50053
```

Use each server module's `--help` if you want to override its provider/model defaults.
