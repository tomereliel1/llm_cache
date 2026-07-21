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

```powershell
python scripts/analyze_trace.py `
  --input datasets/quora_1000.h5 `
  --thresholds 0.7 0.8 0.9 `
  --capacity 1000 `
  --output reports/quora_1000.md
```

The report includes:

- hit rate for each threshold
- miss count
- estimated LLM calls saved
- average similarity of accepted cache hits
- nearest-neighbor similarity statistics
- real example prompt pairs that became cache hits

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
