from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorStoreResult:
    """Result returned by a vector-store similarity lookup.

    Attributes:
        found: Whether a cached entry satisfied the store's similarity threshold.
        prompt: Cached prompt for the matching entry, or an empty string on misses.
        response: Cached response for the matching entry, or an empty string on misses.
        score: Optional provider-specific similarity or distance score.
    """

    found: bool
    prompt: str
    response: str
    score: float | None = None


class IVectorStore(ABC):
    """Interface for semantic-cache storage backends."""

    @abstractmethod
    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        """Find a cached entry similar to an embedding vector.

        Args:
            vector: Prompt embedding vector to search for.

        Returns:
            VectorStoreResult containing the matching entry when one satisfies the
            configured threshold, otherwise a miss result.

        Raises:
            ValueError: If the vector is empty, malformed, or otherwise invalid.
            RuntimeError: If the backing store cannot complete the lookup.
        """

    @abstractmethod
    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        """Store a prompt/response pair and its embedding vector.

        Args:
            prompt: Prompt text associated with the response.
            response: Response text to cache.
            vector: Embedding vector for the prompt.

        Returns:
            Identifier assigned to the stored cache entry.

        Raises:
            ValueError: If the prompt, response, or vector is invalid.
            RuntimeError: If the backing store cannot persist the entry.
        """
