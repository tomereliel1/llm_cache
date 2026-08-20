from abc import ABC, abstractmethod

from llm_cache.vector_store.models import CacheEntryMetadata


class IEvictionPolicy(ABC):
    """Interface for cache eviction policy implementations."""

    @abstractmethod
    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        """Choose which cache entry should be removed.

        Args:
            entries: Metadata for current cache entries.

        Returns:
            Identifier of the entry selected for eviction.

        Raises:
            ValueError: If no entries are available to choose from.
        """
