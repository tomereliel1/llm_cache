# Developer Guide

This guide gives a high-level view of the LLM Semantic Cache implementation for developers
who need to understand, build, test, or extend the project.

For user-facing installation and run instructions, see [USER_GUIDE.md](USER_GUIDE.md).

## High-Level Design

The system is a distributed semantic cache for LLM responses. A prompt flows through these
steps:

```text
prompt
  -> embedding service
  -> vector-store similarity search
       -> cache hit: return stored response
       -> cache miss: call LLM service, store prompt/vector/response, return response
```

The central cache algorithm is implemented in
[orchestrator.py](../src/llm_cache/orchestrator/orchestrator.py). It depends only on three
interfaces:

- `IEmbedder` from [embedding/interface.py](../src/llm_cache/embedding/interface.py)
- `IVectorStore` from [vector_store/interface.py](../src/llm_cache/vector_store/interface.py)
- `ILLMProvider` from [llm/interface.py](../src/llm_cache/llm/interface.py)

This keeps the orchestrator independent from provider choices, gRPC transport, Docker, CLI
parsing, and external services.

## Runtime Architecture

The production-shaped runtime has five main processes:

```text
browser or CLI client
  -> orchestrator gRPC service (:50050)
      -> embedding gRPC service (:50051)
      -> vector-store gRPC service (:50052)
      -> LLM gRPC service (:50053), only on cache misses
```

Each gRPC client implements the same internal interface as the local provider. Because of
that, `CacheOrchestrator` can work with local implementations, test doubles, or remote gRPC
providers without changing the cache algorithm.

## Main Directories

| Path | Purpose |
| --- | --- |
| [src/llm_cache](../src/llm_cache) | Main Python package. |
| [src/llm_cache/orchestrator](../src/llm_cache/orchestrator) | Cache algorithm and public orchestrator gRPC API. |
| [src/llm_cache/embedding](../src/llm_cache/embedding) | Embedding interface, Ollama provider, and embedding gRPC transport. |
| [src/llm_cache/llm](../src/llm_cache/llm) | LLM interface, Ollama/Groq providers, and LLM gRPC transport. |
| [src/llm_cache/vector_store](../src/llm_cache/vector_store) | Vector-store interface, Chroma/in-memory stores, eviction, and gRPC transport. |
| [src/llm_cache/config](../src/llm_cache/config) | Dataclasses, CLI parsing, provider registry, runtime JSON configuration, and validation. |
| [src/llm_cache/factories](../src/llm_cache/factories) | Factory functions that create providers from config objects. |
| [src/llm_cache/health](../src/llm_cache/health) | Health-check protocol and setup-check helpers. |
| [src/llm_cache/web](../src/llm_cache/web) | HTTP web client that sends prompts to the orchestrator gRPC service. |
| [src/llm_cache/test_doubles](../src/llm_cache/test_doubles) | Deterministic stubs and spies for tests and demos. |
| [demos](../demos) | Executable demos for local and mixed gRPC workflows. |
| [configs](../configs) | Example runtime configuration files. |
| [tests](../tests) | Unit and integration tests. |
| [docs](../docs) | User, developer, Docker, logging, tracing, and vector-store documentation. |

A more exhaustive project tree is available in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Core Cache Algorithm

[CacheOrchestrator](../src/llm_cache/orchestrator/orchestrator.py) owns the semantic cache
decision:

```text
1. Validate and trim the prompt.
2. Ask the embedder for a vector.
3. Search the vector store for a similar cached entry.
4. If found, return the stored response with cache_hit=true.
5. If not found, ask the LLM provider for a response.
6. Store prompt, response, and vector in the vector store.
7. Return the generated response with cache_hit=false.
```

The result object is `QueryResult`, which contains:

- `response`: answer returned to the user
- `cache_hit`: whether the answer came from the cache
- `score`: optional similarity score from the vector store

The orchestrator intentionally does not know provider names like `ollama`, `groq`, or
`chroma`. Those choices belong to configuration and factories.

## gRPC Layout

Each distributed component follows the same transport layout:

```text
component/
  interface.py
  providers/ or implementations/
  grpc/
    client.py
    service.py
    server.py
    component.proto
    generated/
```

The roles are:

| File | Role |
| --- | --- |
| `grpc/client.py` | Implements the component interface over a gRPC channel. |
| `grpc/service.py` | Converts protobuf requests into calls on an injected interface implementation. |
| `grpc/server.py` | Parses configuration, constructs the provider, registers the gRPC service, and starts the server. |
| `grpc/*.proto` | Defines the RPC contract. |
| `grpc/generated/` | Checked-in generated protobuf Python bindings. |

Example for the embedding service:

```text
EmbeddingGrpcClient.embed(prompt)
  -> generated protobuf stub
  -> EmbeddingGrpcService.Embed(...)
  -> injected IEmbedder implementation
  -> protobuf response
```

The orchestrator service uses the same pattern, but it is the public API:

```text
OrchestratorGrpcClient.query(prompt)
  -> OrchestratorGrpcService.Query(...)
  -> CacheOrchestrator.query(prompt)
```

## Configuration Flow

Configuration has two layers:

1. Local composition config dataclasses in [app_config.py](../src/llm_cache/config/app_config.py).
2. Runtime JSON sections applied to service and client CLI parsers by
   [runtime_config.py](../src/llm_cache/config/runtime_config.py).

The shared runtime config file supports these top-level sections:

```text
embedding_service
vector_store_service
llm_service
orchestrator_service
web_client
cli_client
```

Each executable reads only its own section. For example:

- `llm_cache.embedding.grpc.server` reads `embedding_service`
- `llm_cache.vector_store.grpc.server` reads `vector_store_service`
- `llm_cache.llm.grpc.server` reads `llm_service`
- `llm_cache.orchestrator.grpc.server` reads `orchestrator_service`
- `main.py web` reads `web_client`
- `main.py cli` reads `cli_client`

Command-line arguments override JSON defaults.

Provider names, model names, defaults, supported vector-store distance functions, and
eviction policy names are centralized in
[provider_options.py](../src/llm_cache/config/provider_options.py). When adding a provider
or model, update that registry first. For more detail, see
[CONFIGURATION_CHANGE_GUIDE.md](CONFIGURATION_CHANGE_GUIDE.md).

## Provider Factories

The factory package translates validated config objects into concrete implementations:

| File | Creates |
| --- | --- |
| [embedding_factory.py](../src/llm_cache/factories/embedding_factory.py) | `OllamaEmbedder` or test embedder. |
| [llm_factory.py](../src/llm_cache/factories/llm_factory.py) | `OllamaLLMProvider`, `GroqLLMProvider`, or test LLM provider. |
| [vector_store_factory.py](../src/llm_cache/factories/vector_store_factory.py) | Chroma, in-memory, hit-stub, or miss-stub vector stores. |
| [eviction_policy_factory.py](../src/llm_cache/factories/eviction_policy_factory.py) | Eviction policy implementations such as LRU. |

The factories are used by local demos and by the provider gRPC servers. The distributed
orchestrator server does not create concrete providers directly; it creates gRPC clients for
the three provider services.

## Vector Store And Eviction

Vector-store code is under [vector_store](../src/llm_cache/vector_store):

- [interface.py](../src/llm_cache/vector_store/interface.py) defines the store contract.
- [models.py](../src/llm_cache/vector_store/models.py) defines cache-entry metadata.
- [implementations/memory.py](../src/llm_cache/vector_store/implementations/memory.py)
  provides an in-process store.
- [implementations/chroma.py](../src/llm_cache/vector_store/implementations/chroma.py)
  provides a Chroma-backed store.
- [eviction/lru.py](../src/llm_cache/vector_store/eviction/lru.py) implements least-recently-used
  eviction.

Similarity behavior depends on the selected provider and distance function. See
[VECTOR_STORE_DISTANCE.md](VECTOR_STORE_DISTANCE.md) before changing thresholds, Chroma
collections, or distance functions.

## Web And CLI Clients

The root [main.py](../main.py) is a convenience entrypoint. It defaults to the web client:

```bash
uv run python main.py web
uv run python main.py cli
```

The web client in [web/main.py](../src/llm_cache/web/main.py):

- serves a simple browser UI
- exposes a `/health` endpoint
- forwards prompts to `OrchestratorGrpcClient`
- displays cache-hit state and recent prompt history

The CLI client in [cli/main.py](../src/llm_cache/cli/main.py):

- starts an interactive prompt loop
- sends each prompt to `OrchestratorGrpcClient`
- prints the response, cache-hit state, and optional similarity score

Both clients depend on the public orchestrator gRPC service rather than calling providers
directly.

## Logging, Tracing, And Health Checks

[logging_config.py](../src/llm_cache/logging_config.py) configures service logs.
[request_context.py](../src/llm_cache/request_context.py) stores request IDs in a context
variable and propagates them through gRPC metadata. This makes it possible to trace one user
prompt through the web client, orchestrator, embedding service, vector store, and LLM service.

For runtime log examples, see [LOGGING.md](LOGGING.md).

The [health](../src/llm_cache/health) package defines reusable health-check types and helpers.
Provider servers can run setup checks before serving, and the orchestrator can verify that its
provider gRPC targets are reachable.

Example:

```bash
uv run python -m llm_cache.orchestrator.grpc.server \
  --embedding-target localhost:50051 \
  --vector-store-target localhost:50052 \
  --llm-target localhost:50053 \
  --check-setup
```

## Tests

Tests live under [tests](../tests):

| Path | Coverage |
| --- | --- |
| [tests/orchestrator](../tests/orchestrator) | Cache hit/miss behavior and orchestrator gRPC API. |
| [tests/embedding](../tests/embedding) | Embedding service, client/server behavior, and integration. |
| [tests/llm](../tests/llm) | LLM service, client/server behavior, and integration. |
| [tests/vector_store](../tests/vector_store) | In-memory store, Chroma store, LRU eviction, and vector-store gRPC service. |
| [tests/test_*.py](../tests) | CLI parsing, runtime config, factories, health, demos, web client, tracing, and wiring. |

Run all tests:

```bash
uv run pytest
```

Run one test file:

```bash
uv run pytest tests/orchestrator/test_cache_orchestrator.py
```

## Build And Quality Tools

The project uses:

- Python 3.12
- `uv` for dependency management
- `pytest` for tests
- `ruff` for formatting and linting
- `grpcio` and `grpcio-tools` for gRPC
- `protobuf` for generated RPC bindings
- `chromadb` for the Chroma vector store
- `ollama` for local embedding and LLM providers
- `groq` for the optional Groq LLM provider
- Docker and Docker Compose for the distributed runtime

Common verification commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
docker compose config
```


## Generated Protobuf Files

Generated protobuf files are checked into the repository under each component's
`grpc/generated/` directory. They are excluded from Ruff formatting and lint rules in
[pyproject.toml](../pyproject.toml).

If a `.proto` contract changes, regenerate the matching Python bindings with
`grpcio-tools`, then run the tests. Keep the generated files committed so users do not need
to regenerate protobuf code before running the project.

## Extending The Project

When adding a new embedding provider:

1. Implement `IEmbedder`.
2. Add the provider and supported models to `provider_options.py`.
3. Update `embedding_factory.py`.
4. Add tests for factory mapping and provider behavior.

When adding a new LLM provider:

1. Implement `ILLMProvider`.
2. Add the provider and supported models to `provider_options.py`.
3. Update `llm_factory.py`.
4. Add tests for configuration, factory mapping, and provider behavior.

When adding a new vector store:

1. Implement `IVectorStore`.
2. Add the provider to `provider_options.py`.
3. Update `vector_store_factory.py`.
4. Decide how capacity and eviction should work.
5. Add tests for search, store, threshold behavior, and gRPC adaptation if exposed remotely.

The main design rule is to keep concrete provider logic outside
`orchestrator/orchestrator.py`. The orchestrator should continue to depend only on the three
interfaces.

## Additional Documentation

- [USER_GUIDE.md](USER_GUIDE.md): installation, configuration, and running instructions.
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): detailed repository tree and module list.
- [DOCKER.md](DOCKER.md): Docker Compose workflow and troubleshooting.
- [LOGGING.md](LOGGING.md): request tracing and cache hit/miss logs.
- [TRACE_ANALYSIS.md](TRACE_ANALYSIS.md): analyzing prompt traces.
- [VECTOR_STORE_DISTANCE.md](VECTOR_STORE_DISTANCE.md): vector distance functions and thresholds.
- [CONFIGURATION_CHANGE_GUIDE.md](CONFIGURATION_CHANGE_GUIDE.md): adding providers, models, and config fields.
