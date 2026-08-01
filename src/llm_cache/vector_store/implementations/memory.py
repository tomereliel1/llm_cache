from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from time import monotonic

from llm_cache.health.health_check_result import HealthCheckResult
from llm_cache.vector_store.eviction.interface import IEvictionPolicy
from llm_cache.vector_store.eviction.lru import LRUEvictionPolicy
from llm_cache.vector_store.interface import IVectorStore, VectorStoreResult
from llm_cache.vector_store.models import CacheEntryMetadata


@dataclass(frozen=True)
class _CacheEntry:
    id: str
    prompt: str
    response: str
    vector: list[float]
    created_at: float
    last_accessed_at: float


class InMemoryVectorStore(IVectorStore):
    """Process-local vector store for development, tests, and simple deployments.

    Entries are kept in memory only and are lost when the process exits. Similarity scores
    use cosine similarity, and capacity is enforced through the configured eviction policy.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_capacity: int = 1000,
        eviction_policy: IEvictionPolicy | None = None,
    ) -> None:
        """Create an in-memory vector store.

        Args:
            similarity_threshold: Minimum cosine similarity required for a cache hit.
            max_capacity: Maximum number of entries retained by the store.
            eviction_policy: Policy used when inserting into a full cache. Defaults to LRU.

        Raises:
            ValueError: If the threshold is outside [0, 1] or capacity is smaller than 1.
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if max_capacity < 1:
            raise ValueError("max_capacity must be at least 1")

        self.similarity_threshold = similarity_threshold
        self.max_capacity = max_capacity
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self._entries: list[_CacheEntry] = []
        self._next_id = 1
        self._lock = RLock()

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        """Find the best cached response for an embedding vector.

        Args:
            vector: Prompt embedding vector to compare against cached entries.

        Returns:
            VectorStoreResult with the best matching entry when it satisfies the
            configured threshold, otherwise a miss result.

        Raises:
            ValueError: If the vector is empty, non-finite, dimensionally incompatible
            with stored entries, or a zero vector.
        """
        self._validate_vector(vector)

        with self._lock:
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
                    score=best_score,
                )

            return VectorStoreResult(found=False, prompt="", response="")

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        """Store a cache entry in memory.

        Args:
            prompt: Prompt text associated with the response.
            response: Response text to cache.
            vector: Embedding vector for the prompt.

        Returns:
            Generated cache entry identifier.

        Raises:
            ValueError: If the prompt, response, or vector is invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")

        self._validate_vector(vector)
        with self._lock:
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

    def health_check(self) -> HealthCheckResult:
        """Report readiness and current cache occupancy.

        Returns:
            Successful HealthCheckResult for the in-memory store.
        """
        with self._lock:
            entry_count = len(self._entries)

        return HealthCheckResult.ok(
            name="vector-store:in-memory",
            message=(
                f"In-memory vector store is ready with {entry_count}/{self.max_capacity} entries"
            ),
        )

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
        """Compute cosine similarity for two validated embedding vectors.

        Args:
            left: First embedding vector.
            right: Second embedding vector.

        Returns:
            Cosine similarity in the range [-1, 1].

        Raises:
            ValueError: If vector dimensions differ or either vector is zero.
        """
        if len(left) != len(right):
            raise ValueError("vectors must have the same dimension")

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            raise ValueError("vectors must not be zero vectors")

        dot_product = sum(
            left_value * right_value for left_value, right_value in zip(left, right, strict=True)
        )
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        """Validate vector shape and numeric values accepted by this store.

        Args:
            vector: Embedding vector to validate.

        Raises:
            ValueError: If the vector is empty, contains non-finite values, or is zero.
        """
        if not vector:
            raise ValueError("vector must not be empty")

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("vector values must be finite numbers")

        if all(value == 0 for value in vector):
            raise ValueError("vectors must not be zero vectors")
