# Configuration Change Guide

Use this guide as context whenever adding or changing supported runtime configuration for `main.py`.
It lists the files that participate in the configuration path so provider/model changes do not get missed.

## Current Configuration Flow

```text
main.py
  -> parse_cli_args()
  -> app_config_from_args()
  -> AppConfig
  -> factories
  -> CacheOrchestrator
```

The orchestrator must stay independent from CLI flags, provider names, concrete provider classes, and model registries.

## Source Of Truth

The central registry is:

```text
src/llm_cache/config/provider_options.py
```

This file owns:

- supported embedding providers and models
- embedding provider default models
- supported LLM providers and models
- LLM provider default models
- supported vector-store providers
- global default provider choices
- default prompt
- default similarity threshold
- `--list-supported-configs` output

Do not duplicate provider/model/default strings in `main.py`, CLI code, or tests unless a test is specifically checking factory examples or concrete behavior.

## Adding A New Model For An Existing Embedding Provider

Example: add another Ollama embedding model.

Required changes:

- Update `SUPPORTED_EMBEDDING_PROVIDERS` in `src/llm_cache/config/provider_options.py`.
- Add the model to the provider's `supported_models` tuple.
- If this model should become the default, update that provider's `default_model`.
- If the global default provider changes, update `DEFAULT_EMBEDDING_PROVIDER`.

Usually no changes are needed in:

- `src/llm_cache/config/cli_args.py`: it reads choices/defaults from the registry.
- `src/llm_cache/factories/embedding_factory.py`: same provider, same concrete implementation.
- `main.py`: it only wires parsed config into factories.
- `src/llm_cache/orchestrator/cache_orchestrator.py`: no provider/model knowledge belongs there.

Tests to consider:

- `tests/test_cli_args.py`: existing parametrized registry tests should cover provider-specific defaults and list output.
- Add a targeted CLI test only if there is new behavior, such as a custom validation rule.
- `tests/test_factories.py`: no new test is needed for a same-provider model unless the factory behavior changes.

Verification commands:

```bash
uv run python main.py --list-supported-configs
uv run python main.py --embedding-provider ollama --embedding-model <new-model>
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Adding A New Model For An Existing LLM Provider

Example: add another Groq or Ollama LLM model.

Required changes:

- Update `SUPPORTED_LLM_PROVIDERS` in `src/llm_cache/config/provider_options.py`.
- Add the model to the provider's `supported_models` tuple.
- If this model should become the default, update that provider's `default_model`.
- If the global default provider changes, update `DEFAULT_LLM_PROVIDER`.

Usually no changes are needed in:

- `src/llm_cache/config/cli_args.py`
- `src/llm_cache/factories/llm_factory.py`
- `main.py`
- `src/llm_cache/orchestrator/cache_orchestrator.py`

Tests to consider:

- `tests/test_cli_args.py`: registry-based tests should cover the model if it is in `supported_models`.
- Add a targeted test only for behavior not already covered by registry-parametrized tests.

Verification commands:

```bash
uv run python main.py --list-supported-configs
uv run python main.py --llm-provider <provider> --llm-model <new-model>
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Adding A New Embedding Provider

Required changes:

- Add a provider entry to `SUPPORTED_EMBEDDING_PROVIDERS` in `src/llm_cache/config/provider_options.py`.
- Add or reuse an implementation under `src/llm_cache/embedding/`.
- Ensure the implementation satisfies `IEmbedder` from `src/llm_cache/embedding/i_embedder.py`.
- Export the implementation from `src/llm_cache/embedding/__init__.py` if tests or callers should import it from the package.
- Update `src/llm_cache/factories/embedding_factory.py` to map the new provider name to the concrete class.
- If the provider needs new config fields, update `EmbeddingConfig` in `src/llm_cache/config/app_config.py`.
- If users need to pass those new fields through CLI, update `src/llm_cache/config/cli_args.py`.

Tests to add or update:

- Add a factory test in `tests/test_factories.py` proving the provider maps to the expected concrete implementation.
- Add missing-config tests for required provider-specific settings.
- Confirm `tests/test_cli_args.py` covers CLI choices, defaults, invalid provider/model combinations, and `--list-supported-configs`.
- Add test doubles only if the new provider needs isolated tests without external services.

Verification commands:

```bash
uv run python main.py --help
uv run python main.py --list-supported-configs
uv run python main.py --embedding-provider <provider> --embedding-model <model>
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Adding A New LLM Provider

Required changes:

- Add a provider entry to `SUPPORTED_LLM_PROVIDERS` in `src/llm_cache/config/provider_options.py`.
- Add or reuse an implementation under `src/llm_cache/llm/`.
- Ensure the implementation satisfies `ILLMProvider` from `src/llm_cache/llm/i_llm_provider.py`.
- Export the implementation from `src/llm_cache/llm/__init__.py` if tests or callers should import it from the package.
- Update `src/llm_cache/factories/llm_factory.py` to map the new provider name to the concrete class.
- If the provider needs API keys, base URLs, or other settings, update `LLMConfig` in `src/llm_cache/config/app_config.py`.
- If users need to pass those settings through CLI, update `src/llm_cache/config/cli_args.py`.

Tests to add or update:

- Add a factory test in `tests/test_factories.py` proving the provider maps to the expected concrete implementation.
- Add tests for required environment variables or config fields.
- Confirm `tests/test_cli_args.py` covers CLI choices, provider-specific defaults, invalid model combinations, and `--list-supported-configs`.

Verification commands:

```bash
uv run python main.py --help
uv run python main.py --list-supported-configs
uv run python main.py --llm-provider <provider> --llm-model <model>
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Adding A New Vector-Store Provider

Required changes:

- Add a provider entry to `SUPPORTED_VECTOR_STORE_PROVIDERS` in `src/llm_cache/config/provider_options.py`.
- Add or reuse an implementation under `src/llm_cache/vector_store/` or `src/llm_cache/test_doubles/`.
- Ensure the implementation satisfies `IVectorStore` from `src/llm_cache/vector_store/i_vector_store.py`.
- Update `src/llm_cache/factories/vector_store_factory.py` to map the new provider name to the concrete class.
- If the provider needs new settings, update `VectorStoreConfig` in `src/llm_cache/config/app_config.py`.
- If users need to pass those settings through CLI, update `src/llm_cache/config/cli_args.py`.

Tests to add or update:

- Add a factory test in `tests/test_factories.py`.
- Add provider-specific behavior tests if the vector store has meaningful behavior beyond construction.
- Confirm `tests/test_cli_args.py` covers CLI choices and `--list-supported-configs`.

Verification commands:

```bash
uv run python main.py --list-supported-configs
uv run python main.py --vector-store-provider <provider>
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Changing Global Defaults

Required changes:

- Update the relevant constant in `src/llm_cache/config/provider_options.py`:
  - `DEFAULT_PROMPT`
  - `DEFAULT_SIMILARITY_THRESHOLD`
  - `DEFAULT_EMBEDDING_PROVIDER`
  - `DEFAULT_LLM_PROVIDER`
  - `DEFAULT_VECTOR_STORE_PROVIDER`
- Ensure the selected default provider exists in the matching supported-provider dictionary.
- Ensure the selected default provider has a valid `default_model` if it is a model provider.

Usually no changes are needed in tests because defaults should be asserted through registry constants.

Verification commands:

```bash
uv run python main.py --help
uv run python main.py --list-supported-configs
uv run python main.py
uv run pytest
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

## Files That Should Usually Stay Unchanged

These files should not need provider/model knowledge for ordinary configuration additions:

- `main.py`: composition root only.
- `src/llm_cache/orchestrator/cache_orchestrator.py`: orchestrator behavior only.
- `src/llm_cache/orchestrator/orchestrator.proto`: not used by the current CLI path.
- `src/llm_cache/embedding/embedding.proto`, `src/llm_cache/llm/llm.proto`, `src/llm_cache/vector_store/vector_store.proto`: only update when the API contract changes.

If one of these files needs a config-related change, double-check that the change is not leaking CLI/provider details into the wrong layer.

## Quick Review Checklist

Before calling the change complete, verify:

- The new provider/model appears in `uv run python main.py --list-supported-configs`.
- `uv run python main.py --help` still shows the right flags and defaults.
- CLI invalid combinations fail before provider construction.
- Factories still own provider-name to concrete-class mapping.
- The orchestrator has no provider-name or concrete-provider imports.
- Tests do not hardcode defaults when they can use registry constants.
- `uv run pytest` passes.
- `uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py` passes.
