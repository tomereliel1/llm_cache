from llm_cache.vector_store.eviction.interface import IEvictionPolicy
from llm_cache.vector_store.models import CacheEntryMetadata


class LRUEvictionPolicy(IEvictionPolicy):
    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        if not entries:
            raise ValueError("entries must not be empty")

        return min(
            entries,
            key=lambda entry: (entry.last_accessed_at, entry.created_at, entry.id),
        ).id
