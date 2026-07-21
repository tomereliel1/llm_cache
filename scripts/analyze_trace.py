from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_cache.config.app_config import EmbeddingConfig, VectorStoreConfig
from llm_cache.factories.embedding_factory import create_embedder
from llm_cache.factories.vector_store_factory import create_vector_store
from llm_cache.llm import ILLMProvider
from llm_cache.orchestrator import CacheOrchestrator
from llm_cache.vector_store import IVectorStore, VectorStoreResult


def _load_optional_dependencies():
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Missing trace-analysis dependencies. Install them with:\n"
            "  python -m pip install -e .[trace]\n"
            "or:\n"
            "  python -m pip install h5py numpy"
        ) from error

    return h5py, np


@dataclass(frozen=True)
class HitExample:
    prompt: str
    cached_prompt: str
    similarity: float | None
    index: int
    cached_index: int


@dataclass(frozen=True)
class SimulationResult:
    backend: str
    threshold: float
    total_prompts: int
    hits: int
    misses: int
    hit_rate: float
    avg_hit_distance: float | None
    best_hit_distance: float | None
    worst_hit_distance: float | None
    p50_nearest_similarity: float | None
    p90_nearest_similarity: float | None
    examples: list[HitExample]


class _TraceLLMProvider(ILLMProvider):
    def __init__(self) -> None:
        self.calls_count = 0

    def generate_answer(self, prompt: str) -> str:
        self.calls_count += 1
        return f"trace-generated-answer-{self.calls_count}"


class _RecordingVectorStore(IVectorStore):
    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store
        self.last_cached_prompt: str | None = None

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        result = self._vector_store.search_similar(vector)
        self.last_cached_prompt = result.prompt if result.found else None
        return result

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        return self._vector_store.store(prompt, response, vector)


def _decode_texts(raw_texts) -> list[str]:
    texts: list[str] = []
    for text in raw_texts:
        if isinstance(text, bytes):
            texts.append(text.decode("utf-8", errors="replace"))
        else:
            texts.append(str(text))
    return texts


def _missing_trace_message(path: Path) -> str:
    return (
        f"Trace file not found: {path}\n"
        "Create it first with:\n"
        "  python scripts/fetch_datasets.py --dataset quora --limit 1000 --model embeddinggemma"
    )


def load_trace_texts(path: Path, max_prompts: int | None) -> list[str]:
    h5py, _ = _load_optional_dependencies()

    if not path.exists():
        raise SystemExit(_missing_trace_message(path))

    with h5py.File(path, "r") as file:
        if "text" not in file:
            raise SystemExit(f"{path} must contain a 'text' dataset")
        texts = _decode_texts(file["text"][:])

    if max_prompts is not None:
        texts = texts[:max_prompts]

    return texts


def load_trace(path: Path, max_prompts: int | None):
    h5py, np = _load_optional_dependencies()

    if not path.exists():
        raise SystemExit(_missing_trace_message(path))

    with h5py.File(path, "r") as file:
        if "normalized_embeddings" in file:
            vectors = file["normalized_embeddings"][:]
        elif "normalized_embeds" in file:
            vectors = file["normalized_embeds"][:]
        else:
            raise SystemExit(
                f"{path} must contain 'normalized_embeddings' or 'normalized_embeds'"
            )

        if "text" not in file:
            raise SystemExit(f"{path} must contain a 'text' dataset")
        texts = _decode_texts(file["text"][:])

    if max_prompts is not None:
        vectors = vectors[:max_prompts]
        texts = texts[:max_prompts]

    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) != len(texts):
        raise SystemExit(
            f"Trace has {len(vectors)} embeddings but {len(texts)} text entries"
        )

    return vectors, texts


def _percentile(np, values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def simulate_cache(
    vectors,
    texts: list[str],
    *,
    threshold: float,
    capacity: int,
    examples_count: int,
) -> SimulationResult:
    _, np = _load_optional_dependencies()

    cache: OrderedDict[int, object] = OrderedDict()
    hits = 0
    misses = 0
    hit_similarities: list[float] = []
    nearest_similarities: list[float] = []
    examples: list[HitExample] = []

    for index, vector in enumerate(vectors):
        if cache:
            cached_indices = list(cache)
            cached_vectors = np.asarray([cache[cached_index] for cached_index in cached_indices])
            similarities = cached_vectors @ vector
            best_position = int(np.argmax(similarities))
            best_similarity = float(similarities[best_position])
            best_index = cached_indices[best_position]
            nearest_similarities.append(best_similarity)
        else:
            best_similarity = -1.0
            best_index = -1

        if best_similarity >= threshold:
            hits += 1
            hit_similarities.append(best_similarity)
            cache.move_to_end(best_index)
            if len(examples) < examples_count:
                examples.append(
                    HitExample(
                        prompt=texts[index],
                        cached_prompt=texts[best_index],
                        similarity=best_similarity,
                        index=index,
                        cached_index=best_index,
                    )
                )
        else:
            misses += 1
            cache[index] = vector
            if len(cache) > capacity:
                cache.popitem(last=False)

    total = hits + misses
    hit_distances = [1 - similarity for similarity in hit_similarities]
    return SimulationResult(
        backend="precomputed-vectors",
        threshold=threshold,
        total_prompts=total,
        hits=hits,
        misses=misses,
        hit_rate=hits / total if total else 0,
        avg_hit_distance=(sum(hit_distances) / len(hit_distances) if hit_distances else None),
        best_hit_distance=min(hit_distances) if hit_distances else None,
        worst_hit_distance=max(hit_distances) if hit_distances else None,
        p50_nearest_similarity=_percentile(np, nearest_similarities, 50),
        p90_nearest_similarity=_percentile(np, nearest_similarities, 90),
        examples=examples,
    )


def run_project_vector_store(
    texts: list[str],
    *,
    threshold: float,
    capacity: int,
    examples_count: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_base_url: str | None,
    vector_store_provider: str,
) -> SimulationResult:
    embedder = create_embedder(
        EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            base_url=embedding_base_url,
        )
    )

    with TemporaryDirectory() as temp_dir:
        vector_store = _RecordingVectorStore(
            create_vector_store(
                VectorStoreConfig(
                    provider=vector_store_provider,
                    similarity_threshold=threshold,
                    persist_path=temp_dir,
                    max_capacity=capacity,
                    eviction_policy="lru",
                    persistent=False,
                )
            )
        )
        llm_provider = _TraceLLMProvider()
        orchestrator = CacheOrchestrator(
            embedder=embedder,
            llm_provider=llm_provider,
            vector_store=vector_store,
        )

        hits = 0
        hit_scores: list[float] = []
        examples: list[HitExample] = []

        for index, prompt in enumerate(texts):
            print(f"Project replay prompt {index + 1}/{len(texts)}")
            result = orchestrator.query(prompt)
            if result.cache_hit:
                hits += 1
                if result.score is not None:
                    hit_scores.append(result.score)
                if len(examples) < examples_count:
                    examples.append(
                        HitExample(
                            prompt=prompt,
                            cached_prompt=vector_store.last_cached_prompt or "",
                            similarity=result.score,
                            index=index,
                            cached_index=-1,
                        )
                    )

        misses = llm_provider.calls_count

    total = hits + misses
    return SimulationResult(
        backend=f"project-orchestrator:{vector_store_provider}",
        threshold=threshold,
        total_prompts=total,
        hits=hits,
        misses=misses,
        hit_rate=hits / total if total else 0,
        avg_hit_distance=(sum(hit_scores) / len(hit_scores) if hit_scores else None),
        best_hit_distance=min(hit_scores) if hit_scores else None,
        worst_hit_distance=max(hit_scores) if hit_scores else None,
        p50_nearest_similarity=None,
        p90_nearest_similarity=None,
        examples=examples,
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_report(
    *,
    trace_path: Path,
    capacity: int,
    results: list[SimulationResult],
) -> str:
    lines: list[str] = [
        "# Trace Analysis Report",
        "",
        f"- Trace file: `{trace_path}`",
        f"- Prompts analyzed: {results[0].total_prompts if results else 0}",
        f"- Backend: `{results[0].backend if results else 'n/a'}`",
        f"- Cache capacity: {capacity}",
        "- Eviction policy: LRU",
        "",
        "## Threshold Sweep",
        "",
        (
            "| threshold | hits saved | misses | hit rate | avg hit distance | "
            "best hit distance | worst hit distance |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.threshold:.2f} | "
            f"{result.hits} | "
            f"{result.misses} | "
            f"{_format_percent(result.hit_rate)} | "
            f"{_format_optional_float(result.avg_hit_distance)} | "
            f"{_format_optional_float(result.best_hit_distance)} | "
            f"{_format_optional_float(result.worst_hit_distance)} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `hit rate` estimates how many LLM calls the semantic cache could avoid.",
            "- `hits saved` is the number of prompts served from cache instead of the LLM.",
            (
                "- `avg hit distance` describes the average distance of accepted cache hits. "
                "Lower is better."
            ),
            (
                "- `best hit distance` is the closest accepted match; `worst hit distance` "
                "is the farthest match still accepted by the threshold."
            ),
            "- Higher thresholds usually reduce false hits, but also reduce cache reuse.",
            "",
        ]
    )

    if results:
        best = max(results, key=lambda result: result.hit_rate)
        lines.extend(
            [
                "## Example Cache Hits",
                "",
                f"Examples below use threshold `{best.threshold:.2f}`.",
                "",
            ]
        )
        if not best.examples:
            lines.append("No cache-hit examples were found at this threshold.")
        for number, example in enumerate(best.examples, start=1):
            similarity = _format_optional_float(example.similarity)
            lines.extend(
                [
                    f"### Example {number}",
                    "",
                    f"- score: `{similarity}`",
                    f"- prompt index: `{example.index}`",
                    "",
                    "Prompt:",
                    "",
                    f"> {example.prompt}",
                    "",
                    "Nearest cached prompt:",
                    "",
                    f"> {example.cached_prompt}",
                    "",
                ]
            )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a saved embedding trace and simulate semantic cache behavior."
    )
    parser.add_argument("--input", required=True, help="Input .h5 trace file")
    parser.add_argument(
        "--backend",
        choices=("precomputed-vectors", "project-vector-store"),
        default="precomputed-vectors",
        help=(
            "Analysis backend. precomputed-vectors is fast and uses saved embeddings. "
            "project-vector-store replays prompts through CacheOrchestrator and "
            "the real project vector store. "
            "Default: precomputed-vectors"
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.7, 0.8, 0.9],
        help="Similarity thresholds to test. Default: 0.7 0.8 0.9",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=1000,
        help="Maximum cache entries to keep during simulation. Default: 1000",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Analyze at most this many prompts from the trace.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of cache-hit examples to include. Default: 5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown report path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="ollama",
        help="Embedder provider for --backend project-vector-store. Default: ollama",
    )
    parser.add_argument(
        "--embedding-model",
        default="embeddinggemma",
        help="Embedder model for --backend project-vector-store. Default: embeddinggemma",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="Optional embedder base URL for --backend project-vector-store.",
    )
    parser.add_argument(
        "--vector-store-provider",
        choices=("in-memory", "chroma"),
        default="chroma",
        help="Vector store provider for --backend project-vector-store. Default: chroma",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.capacity < 1:
        parser.error("--capacity must be at least 1")
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be at least 1")
    if args.examples < 0:
        parser.error("--examples must be at least 0")
    for threshold in args.thresholds:
        if not 0 <= threshold <= 1:
            parser.error("--thresholds values must be between 0 and 1")

    trace_path = Path(args.input)
    if args.backend == "precomputed-vectors":
        vectors, texts = load_trace(trace_path, args.max_prompts)
        results = [
            simulate_cache(
                vectors,
                texts,
                threshold=threshold,
                capacity=args.capacity,
                examples_count=args.examples,
            )
            for threshold in args.thresholds
        ]
    else:
        texts = load_trace_texts(trace_path, args.max_prompts)
        results = [
            run_project_vector_store(
                texts,
                threshold=threshold,
                capacity=args.capacity,
                examples_count=args.examples,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                embedding_base_url=args.embedding_base_url,
                vector_store_provider=args.vector_store_provider,
            )
            for threshold in args.thresholds
        ]
    report = build_report(trace_path=trace_path, capacity=args.capacity, results=results)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Saved report to {output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
