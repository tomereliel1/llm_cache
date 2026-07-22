# Trace analysis

This workflow helps evaluate the semantic cache on real prompt traces without
calling an expensive LLM for every miss. It downloads prompts, embeds them, and
then replays the prompt sequence through our cache code.

## Install dependencies

Recommended setup with `uv`:

```bash
uv sync --extra trace
```

If you are not using `uv`, install the optional trace dependencies with pip:

```powershell
python -m pip install -e .[trace]
```

This installs the Hugging Face dataset and HDF5 dependencies used to create and
read trace files.

## Fetch a trace

Start with a small trace so the first run is quick:

PowerShell:

```powershell
python scripts/fetch_datasets.py --dataset quora --limit 1000
```

Bash:

```bash
uv run python scripts/fetch_datasets.py --dataset quora --limit 1000
```

This creates:

```text
datasets/quora_1000.h5
```

The `datasets/` directory is ignored by Git because trace files can be large
and should be regenerated locally by each developer.

Supported datasets:

- `quora`
- `stackoverflow`
- `eli5`
- `natural-questions`
- `msmarco`

The `quora` option uses the Parquet-backed
`sentence-transformers/quora-duplicates` dataset because recent Hugging Face
`datasets` versions no longer load the older script-backed `quora` dataset.

## Analyze the trace

The analyzer replays the prompt texts through our embedder, orchestrator, and
vector store:

PowerShell:

```powershell
python scripts/analyze_trace.py `
  --input datasets/quora_1000.h5 `
  --vector-store-provider chroma `
  --embedding-provider ollama `
  --embedding-model embeddinggemma `
  --thresholds 0.8 `
  --capacity 1000 `
  --output reports/quora_1000.md
```

Bash:

```bash
uv run python scripts/analyze_trace.py \
  --input datasets/quora_1000.h5 \
  --vector-store-provider chroma \
  --embedding-provider ollama \
  --embedding-model embeddinggemma \
  --thresholds 0.8 \
  --capacity 1000 \
  --output reports/quora_1000.md
```

This calls our project code:

```text
OllamaEmbedder -> CacheOrchestrator -> ChromaVectorStore.search_similar/store
```

It uses a fake trace LLM for cache misses because the report only needs to know
whether the cache would avoid an LLM call. The fake LLM makes the run cheaper and
more repeatable while still testing the embedder, orchestrator, and vector store.

The report includes:

- hit rate for each threshold
- miss count
- estimated LLM calls saved
- average distance of accepted cache hits
- best and worst accepted hit distance
- example prompt pairs that became cache hits

Hit/miss counts and hit scores come from our project implementation. With Chroma,
the score is Chroma's returned distance. Lower is better.

The threshold follows the vector store's own behavior. For the in-memory store,
a hit means `similarity >= threshold`. For Chroma, a hit means
`distance <= threshold`.

The `reports/` directory is also ignored by Git. Reports are experiment outputs,
not source files, so do not commit generated files such as
`reports/quora_1000.md`.

## What to say in the final submission

Useful claims are:

- Higher thresholds reduce risky cache hits but also reduce reuse.
- Lower thresholds save more LLM calls but may accept weaker semantic matches.
- Cache capacity matters because old entries are evicted with LRU.
- Real datasets contain semantic repetition, so a vector cache can avoid repeated
  LLM calls even when prompts are not textually identical.

Example wording:

> We evaluated the cache on a Quora prompt trace. At threshold 0.8 with LRU
> capacity 1000, our cache replay measured a cache hit rate of X%, meaning X%
> of LLM calls could be avoided. The accepted Chroma hits had average distance
> Y, where lower distance means a closer semantic match.

## Notes

The fetch script only downloads prompt text. The analysis script embeds those
prompts through the configured project embedder.

With the default Ollama embedder, make sure Ollama is running and the embedding
model is available before running `analyze_trace.py`:

```powershell
ollama serve
ollama pull embeddinggemma
```
