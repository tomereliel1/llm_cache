from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic

from llm_cache.vector_store.cache_entry_metadata import CacheEntryMetadata
from llm_cache.vector_store.i_eviction_policy import IEvictionPolicy
from llm_cache.vector_store.i_vector_store import IVectorStore, VectorStoreResult
from llm_cache.vector_store.lru_eviction_policy import LRUEvictionPolicy


@dataclass(frozen=True)
class _CacheEntry:
    id: str
    prompt: str
    response: str
    vector: list[float]
    created_at: float
    last_accessed_at: float


class InMemoryVectorStore(IVectorStore):
    """Simple local vector store using cosine similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_capacity: int = 1000,
        eviction_policy: IEvictionPolicy | None = None,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if max_capacity < 1:
            raise ValueError("max_capacity must be at least 1")

        self.similarity_threshold = similarity_threshold
        self.max_capacity = max_capacity
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
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
            self._touch(best_entry.id)
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
        self._evict_if_needed()

        entry_id = f"entry-{self._next_id}"
        self._next_id += 1
        now = monotonic()
        self._entries.append(
            _CacheEntry(
                id=entry_id,
                prompt=prompt.strip(),
                response=response,
                vector=list(vector),
                created_at=now,
                last_accessed_at=now,
            )
        )
        return entry_id

    def _touch(self, entry_id: str) -> None:
        now = monotonic()
        self._entries = [
            _CacheEntry(
                id=entry.id,
                prompt=entry.prompt,
                response=entry.response,
                vector=entry.vector,
                created_at=entry.created_at,
                last_accessed_at=now if entry.id == entry_id else entry.last_accessed_at,
            )
            for entry in self._entries
        ]

    def _evict_if_needed(self) -> None:
        if len(self._entries) < self.max_capacity:
            return

        victim_id = self.eviction_policy.choose_victim(
            [
                CacheEntryMetadata(
                    id=entry.id,
                    created_at=entry.created_at,
                    last_accessed_at=entry.last_accessed_at,
                )
                for entry in self._entries
            ]
        )
        self._entries = [entry for entry in self._entries if entry.id != victim_id]

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
