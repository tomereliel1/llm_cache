from abc import ABC, abstractmethod

from llm_cache.vector_store.cache_entry_metadata import CacheEntryMetadata


class IEvictionPolicy(ABC):
    @abstractmethod
    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        """Return the id of the cache entry to evict."""
