from llm_cache.vector_store.cache_entry_metadata import CacheEntryMetadata
from llm_cache.vector_store.i_eviction_policy import IEvictionPolicy


class LRUEvictionPolicy(IEvictionPolicy):
    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        if not entries:
            raise ValueError("entries must not be empty")

        return min(entries, key=lambda entry: entry.last_accessed_at).id
