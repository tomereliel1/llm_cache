import pytest

from llm_cache.vector_store import InMemoryVectorStore


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


def test_store_returns_incrementing_entry_ids() -> None:
    vector_store = InMemoryVectorStore()

    first_id = vector_store.store(prompt="first", response="first response", vector=[1.0])
    second_id = vector_store.store(prompt="second", response="second response", vector=[1.0])

    assert first_id == "entry-1"
    assert second_id == "entry-2"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_rejects_invalid_similarity_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        InMemoryVectorStore(similarity_threshold=threshold)


def test_rejects_dimension_mismatch() -> None:
    vector_store = InMemoryVectorStore()
    vector_store.store(prompt="prompt", response="response", vector=[1.0, 0.0])

    with pytest.raises(ValueError, match="same dimension"):
        vector_store.search_similar([1.0])


def test_rejects_zero_vectors() -> None:
    vector_store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="zero vectors"):
        vector_store.store(prompt="prompt", response="response", vector=[0.0, 0.0])
