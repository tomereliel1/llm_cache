import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_cache.vector_store import CacheEntryMetadata, IEvictionPolicy, InMemoryVectorStore
from llm_cache.vector_store.eviction.lru import LRUEvictionPolicy


class _FirstEntryEvictionPolicy(IEvictionPolicy):
    def choose_victim(self, entries: list[CacheEntryMetadata]) -> str:
        return entries[0].id


def test_empty_store_returns_miss() -> None:
    vector_store = InMemoryVectorStore()

    result = vector_store.search_similar([1.0, 0.0])

    assert result.found is False
    assert result.response == ""


def test_store_and_search_returns_cached_response_above_threshold() -> None:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    vector_store.store(
        prompt="What is semantic caching?",
        response="cached response",
        vector=[1.0, 0.0],
    )

    result = vector_store.search_similar([0.9, 0.1])

    assert result.found is True
    assert result.prompt == "What is semantic caching?"
    assert result.response == "cached response"
    assert result.score == pytest.approx(0.9938837)


def test_search_returns_miss_below_threshold() -> None:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    vector_store.store(
        prompt="What is semantic caching?",
        response="cached response",
        vector=[1.0, 0.0],
    )

    result = vector_store.search_similar([0.0, 1.0])

    assert result.found is False
    assert result.response == ""


def test_search_returns_nearest_cached_response() -> None:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    vector_store.store(prompt="first", response="first response", vector=[1.0, 0.0])
    vector_store.store(prompt="second", response="second response", vector=[0.0, 1.0])

    result = vector_store.search_similar([0.1, 0.9])

    assert result.found is True
    assert result.prompt == "second"
    assert result.response == "second response"


def test_search_uses_first_inserted_entry_for_equal_similarity_tie() -> None:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8)
    vector_store.store(prompt="first", response="first response", vector=[1.0, 0.0])
    vector_store.store(prompt="second", response="second response", vector=[1.0, 0.0])

    result = vector_store.search_similar([1.0, 0.0])

    assert result.found is True
    assert result.prompt == "first"
    assert result.response == "first response"


def test_store_returns_incrementing_entry_ids() -> None:
    vector_store = InMemoryVectorStore()

    first_id = vector_store.store(prompt="first", response="first response", vector=[1.0])
    second_id = vector_store.store(prompt="second", response="second response", vector=[1.0])

    assert first_id == "entry-1"
    assert second_id == "entry-2"


def test_store_evicts_least_recently_used_entry_when_capacity_is_full() -> None:
    vector_store = InMemoryVectorStore(similarity_threshold=0.8, max_capacity=2)
    vector_store.store(prompt="first", response="first response", vector=[1.0, 0.0])
    vector_store.store(prompt="second", response="second response", vector=[0.0, 1.0])

    vector_store.search_similar([1.0, 0.0])
    vector_store.store(prompt="third", response="third response", vector=[0.7, 0.7])

    first_result = vector_store.search_similar([1.0, 0.0])
    second_result = vector_store.search_similar([0.0, 1.0])
    third_result = vector_store.search_similar([0.7, 0.7])

    assert first_result.found is True
    assert first_result.response == "first response"
    assert second_result.found is False
    assert third_result.found is True
    assert third_result.response == "third response"


def test_store_uses_injected_eviction_policy_when_capacity_is_full() -> None:
    vector_store = InMemoryVectorStore(
        similarity_threshold=0.8,
        max_capacity=2,
        eviction_policy=_FirstEntryEvictionPolicy(),
    )
    vector_store.store(prompt="first", response="first response", vector=[1.0, 0.0])
    vector_store.store(prompt="second", response="second response", vector=[0.0, 1.0])

    vector_store.search_similar([1.0, 0.0])
    vector_store.store(prompt="third", response="third response", vector=[0.7, 0.7])

    first_result = vector_store.search_similar([1.0, 0.0])
    second_result = vector_store.search_similar([0.0, 1.0])

    assert first_result.found is False
    assert second_result.found is True
    assert second_result.response == "second response"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_rejects_invalid_similarity_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        InMemoryVectorStore(similarity_threshold=threshold)


def test_rejects_invalid_max_capacity() -> None:
    with pytest.raises(ValueError, match="max_capacity"):
        InMemoryVectorStore(max_capacity=0)


def test_health_check_is_healthy_and_does_not_touch_entries() -> None:
    vector_store = InMemoryVectorStore()
    vector_store.store(prompt="prompt", response="response", vector=[1.0])
    entries_before = list(vector_store._entries)

    result = vector_store.health_check()

    assert result.healthy is True
    assert result.name == "vector-store:in-memory"
    assert vector_store._entries == entries_before


def test_rejects_dimension_mismatch() -> None:
    vector_store = InMemoryVectorStore()
    vector_store.store(prompt="prompt", response="response", vector=[1.0, 0.0])

    with pytest.raises(ValueError, match="same dimension"):
        vector_store.search_similar([1.0])


def test_rejects_zero_vectors() -> None:
    vector_store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="zero vectors"):
        vector_store.store(prompt="prompt", response="response", vector=[0.0, 0.0])


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_vectors(invalid_value: float) -> None:
    vector_store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="finite"):
        vector_store.store(prompt="prompt", response="response", vector=[1.0, invalid_value])


def test_lru_eviction_tie_breaks_by_created_at_then_id() -> None:
    policy = LRUEvictionPolicy()

    victim = policy.choose_victim(
        [
            CacheEntryMetadata(id="entry-2", created_at=20.0, last_accessed_at=100.0),
            CacheEntryMetadata(id="entry-1", created_at=10.0, last_accessed_at=100.0),
        ]
    )

    assert victim == "entry-1"


def test_concurrent_writes_do_not_exceed_configured_capacity() -> None:
    vector_store = InMemoryVectorStore(max_capacity=3, similarity_threshold=1.0)

    def store(index: int) -> None:
        vector_store.store(
            prompt=f"prompt {index}",
            response=f"response {index}",
            vector=[float(index + 1), 1.0],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store, range(50)))

    assert len(vector_store._entries) == 3
    assert len({entry.id for entry in vector_store._entries}) == 3
