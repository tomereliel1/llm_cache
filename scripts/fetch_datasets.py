from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _load_optional_dependencies():
    try:
        import h5py

        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Missing trace-analysis dependencies. Install them with:\n"
            "  python -m pip install -e .[trace]\n"
            "or:\n"
            "  python -m pip install datasets h5py"
        ) from error

    return h5py, load_dataset


def _clean_texts(texts: Sequence[str], limit: int) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for text in texts:
        value = str(text).replace("\x00", "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
        if len(clean) >= limit:
            break
    return clean


def _build_stackoverflow(load_dataset, limit: int) -> list[str]:
    dataset = load_dataset("pacovaldez/stackoverflow-questions")
    texts: list[str] = []
    for split in ("train", "validation", "test"):
        if split in dataset:
            texts.extend(dataset[split]["title"])
            if len(texts) >= limit:
                break
    return _clean_texts(texts, limit)


def _build_quora(load_dataset, limit: int) -> list[str]:
    dataset = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
    texts: list[str] = []
    for row in dataset:
        texts.extend([row["anchor"], row["positive"]])
        if len(texts) >= limit * 2:
            break
    return _clean_texts(texts, limit)


def _build_eli5(load_dataset, limit: int) -> list[str]:
    dataset = load_dataset("sentence-transformers/eli5", trust_remote_code=True)
    return _clean_texts(dataset["train"]["question"], limit)


def _build_natural_questions(load_dataset, limit: int) -> list[str]:
    dataset = load_dataset("nq_open")
    texts: list[str] = []
    for split in ("train", "validation"):
        if split in dataset:
            texts.extend(dataset[split]["question"])
            if len(texts) >= limit:
                break
    return _clean_texts(texts, limit)


def _build_msmarco(load_dataset, limit: int) -> list[str]:
    dataset = load_dataset("ms_marco", "v2.1")
    texts: list[str] = []
    for split in ("train", "validation", "test"):
        if split in dataset:
            texts.extend(dataset[split]["query"])
            if len(texts) >= limit:
                break
    return _clean_texts(texts, limit)


DATASETS: dict[str, Callable[[Any, int], list[str]]] = {
    "stackoverflow": _build_stackoverflow,
    "quora": _build_quora,
    "eli5": _build_eli5,
    "natural-questions": _build_natural_questions,
    "msmarco": _build_msmarco,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a prompt trace from Hugging Face and save it as HDF5."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="quora",
        help="Dataset trace to download. Default: quora",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of unique prompts to keep. Default: 1000",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .h5 path. Default: datasets/<dataset>_<limit>.h5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    output = Path(args.output or f"datasets/{args.dataset}_{args.limit}.h5")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output}. Use --overwrite to replace it.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "100")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "100")

    h5py, load_dataset = _load_optional_dependencies()

    print(f"Loading dataset: {args.dataset}")
    texts = DATASETS[args.dataset](load_dataset, args.limit)
    if not texts:
        raise SystemExit(f"No prompts were loaded from dataset {args.dataset!r}")

    print(f"Loaded {len(texts)} prompts")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        file.create_dataset("text", data=texts, dtype=string_dtype)
        file.attrs["dataset"] = args.dataset

    print(f"Saved trace to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
