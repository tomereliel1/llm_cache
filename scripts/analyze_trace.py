from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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
    similarity: float
    index: int
    cached_index: int


@dataclass(frozen=True)
class SimulationResult:
    threshold: float
    total_prompts: int
    hits: int
    misses: int
    hit_rate: float
    avg_hit_similarity: float | None
    avg_hit_distance: float | None
    p50_nearest_similarity: float | None
    p90_nearest_similarity: float | None
    examples: list[HitExample]


def _decode_texts(raw_texts) -> list[str]:
    texts: list[str] = []
    for text in raw_texts:
        if isinstance(text, bytes):
            texts.append(text.decode("utf-8", errors="replace"))
        else:
            texts.append(str(text))
    return texts


def load_trace(path: Path, max_prompts: int | None):
    h5py, np = _load_optional_dependencies()

    if not path.exists():
        raise SystemExit(
            f"Trace file not found: {path}\n"
            "Create it first with:\n"
            "  python scripts/fetch_datasets.py --dataset quora --limit 1000 --model embeddinggemma"
        )

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
    return SimulationResult(
        threshold=threshold,
        total_prompts=total,
        hits=hits,
        misses=misses,
        hit_rate=hits / total if total else 0,
        avg_hit_similarity=(
            sum(hit_similarities) / len(hit_similarities) if hit_similarities else None
        ),
        avg_hit_distance=(
            1 - (sum(hit_similarities) / len(hit_similarities))
            if hit_similarities
            else None
        ),
        p50_nearest_similarity=_percentile(np, nearest_similarities, 50),
        p90_nearest_similarity=_percentile(np, nearest_similarities, 90),
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
        f"- Cache capacity: {capacity}",
        "- Eviction policy: LRU",
        "",
        "## Threshold Sweep",
        "",
        (
            "| threshold | hits saved | misses | hit rate | avg hit similarity | "
            "avg hit distance | p50 nearest | p90 nearest |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.threshold:.2f} | "
            f"{result.hits} | "
            f"{result.misses} | "
            f"{_format_percent(result.hit_rate)} | "
            f"{_format_optional_float(result.avg_hit_similarity)} | "
            f"{_format_optional_float(result.avg_hit_distance)} | "
            f"{_format_optional_float(result.p50_nearest_similarity)} | "
            f"{_format_optional_float(result.p90_nearest_similarity)} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `hit rate` estimates how many LLM calls the semantic cache could avoid.",
            "- `avg hit similarity` shows how close accepted cache hits were.",
            (
                "- `avg hit distance` is `1 - similarity`, so lower values mean "
                "closer accepted hits."
            ),
            (
                "- `p50 nearest` and `p90 nearest` describe how much semantic reuse "
                "exists in the trace."
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
            lines.extend(
                [
                    f"### Example {number}",
                    "",
                    f"- similarity: `{example.similarity:.3f}`",
                    f"- prompt index: `{example.index}`",
                    f"- cached prompt index: `{example.cached_index}`",
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
