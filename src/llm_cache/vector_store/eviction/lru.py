from llm_cache.vector_store.eviction.interface import IEvictionPolicy
from llm_cache.vector_store.models import CacheEntryMetadata


class LRUEvictionPolicy(IEvictionPolicy):
    """Eviction policy that removes the least recently accessed cache entry."""

    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        """Choose the least recently accessed entry for eviction.

        Args:
            entries: Metadata for current cache entries.

        Returns:
            Identifier of the entry selected for eviction.

        Raises:
            ValueError: If entries is empty.
        """
        if not entries:
            raise ValueError("entries must not be empty")

        return min(
            entries,
            key=lambda entry: (entry.last_accessed_at, entry.created_at, entry.id),
        ).id
