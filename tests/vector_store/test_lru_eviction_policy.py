import pytest

from llm_cache.vector_store import CacheEntryMetadata, LRUEvictionPolicy


def test_choose_victim_returns_least_recently_used_entry_id() -> None:
    policy = LRUEvictionPolicy()
    entries = [
        CacheEntryMetadata(id="newer", created_at=1.0, last_accessed_at=3.0),
        CacheEntryMetadata(id="older", created_at=2.0, last_accessed_at=1.0),
    ]

    victim_id = policy.choose_victim(entries)

    assert victim_id == "older"


def test_choose_victim_rejects_empty_entries() -> None:
    policy = LRUEvictionPolicy()

    with pytest.raises(ValueError, match="entries"):
        policy.choose_victim([])
