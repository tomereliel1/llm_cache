from __future__ import annotations

import math
from dataclasses import dataclass

from llm_cache.vector_store.i_vector_store import IVectorStore, VectorStoreResult


@dataclass(frozen=True)
class _CacheEntry:
    id: str
    prompt: str
    response: str
    vector: list[float]


class InMemoryVectorStore(IVectorStore):
    """Simple local vector store using cosine similarity."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.similarity_threshold = similarity_threshold
        self._entries: list[_CacheEntry] = []
        self._next_id = 1

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        self._validate_vector(vector)

        if not self._entries:
            return VectorStoreResult(found=False, prompt="", response="")

        best_entry = self._entries[0]
        best_score = self._cosine_similarity(vector, best_entry.vector)

        for entry in self._entries[1:]:
            score = self._cosine_similarity(vector, entry.vector)
            if score > best_score:
                best_entry = entry
                best_score = score

        if best_score >= self.similarity_threshold:
            return VectorStoreResult(
                found=True,
                prompt=best_entry.prompt,
                response=best_entry.response,
            )

        return VectorStoreResult(found=False, prompt="", response="")

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")

        self._validate_vector(vector)

        entry_id = f"entry-{self._next_id}"
        self._next_id += 1
        self._entries.append(
            _CacheEntry(
                id=entry_id,
                prompt=prompt.strip(),
                response=response,
                vector=list(vector),
            )
        )
        return entry_id

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have the same dimension")

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            raise ValueError("vectors must not be zero vectors")

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")

        if all(value == 0 for value in vector):
            raise ValueError("vectors must not be zero vectors")
