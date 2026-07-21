# Trace analysis

This workflow helps evaluate the semantic cache on real prompt traces without
calling an LLM. It downloads prompts, embeds them, then simulates cache
hit/miss decisions over the trace.

## Install dependencies

```powershell
python -m pip install -e .[trace]
```

This installs the Hugging Face dataset and HDF5 dependencies. It does not install
`sentence-transformers` or Torch by default; the fetch script uses the project's
existing Ollama embedding provider.

## Fetch a trace

Start with a small trace so the first run is quick:

```powershell
python scripts/fetch_datasets.py --dataset quora --limit 1000 --model embeddinggemma
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

Fast offline simulation using the embeddings saved in the trace file:

```powershell
python scripts/analyze_trace.py `
  --input datasets/quora_1000.h5 `
  --thresholds 0.7 0.8 0.9 `
  --capacity 1000 `
  --output reports/quora_1000.md
```

Replay the same prompt texts through the actual project embedder and vector
store implementation:

```powershell
python scripts/analyze_trace.py `
  --input datasets/quora_1000.h5 `
  --backend project-vector-store `
  --vector-store-provider chroma `
  --embedding-provider ollama `
  --embedding-model embeddinggemma `
  --thresholds 0.8 `
  --capacity 1000 `
  --output reports/quora_1000.md
```

The project-vector-store backend calls the real project code:

```text
OllamaEmbedder -> CacheOrchestrator -> ChromaVectorStore.search_similar/store
```

This is slower than the precomputed-vector backend because it embeds every prompt
again. It is useful when you want to demonstrate that the actual orchestrator,
embedder, and vector store implementation work on the dataset prompts.

The report includes:

- hit rate for each threshold
- miss count
- estimated LLM calls saved
- average distance of accepted cache hits
- best and worst accepted hit distance
- real example prompt pairs that became cache hits

In `project-vector-store` mode, hit/miss counts and hit scores come from the
actual project implementation.

Distance meaning depends on the selected backend:

- `precomputed-vectors`: reports cosine similarity and `1 - similarity` distance.
- `project-vector-store` with `chroma`: reports Chroma's returned distance. Lower is better.

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
> capacity 1000, the simulator estimated a cache hit rate of X%, meaning X% of
> LLM calls could be avoided. The accepted hits had average similarity Y, which
> shows that many prompts were semantically close even when not identical.

## Notes

The analysis script uses saved embeddings and does not call the configured LLM.
That makes it fast, reproducible, and cheap to run during experiments.

The fetch script does call the embedding provider. With the default Ollama provider,
make sure Ollama is running and the embedding model is available:

```powershell
ollama serve
ollama pull embeddinggemma
```

If you explicitly want to use Sentence Transformers instead, run:

```powershell
python -m pip install sentence-transformers
python scripts/fetch_datasets.py --dataset quora --limit 1000 `
  --embedding-provider sentence-transformers --model all-MiniLM-L6-v2
```

On Windows, installing Torch through Sentence Transformers may fail unless long
paths are enabled. The Ollama path avoids that dependency.
