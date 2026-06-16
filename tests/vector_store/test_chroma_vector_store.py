import pytest

from llm_cache.vector_store import ChromaVectorStore, LRUEvictionPolicy


def test_empty_chroma_store_returns_miss(tmp_path) -> None:
    vector_store = ChromaVectorStore(persist_path=str(tmp_path))

    result = vector_store.search_similar([1.0, 0.0])

    assert result.found is False
    assert result.response == ""


def test_store_and_search_returns_cached_response_within_threshold(tmp_path) -> None:
    vector_store = ChromaVectorStore(similarity_threshold=0.8, persist_path=str(tmp_path))
    vector_store.store(
        prompt="What is semantic caching?",
        response="cached response",
        vector=[1.0, 0.0],
    )

    result = vector_store.search_similar([0.9, 0.1])

    assert result.found is True
    assert result.prompt == "What is semantic caching?"
    assert result.response == "cached response"


def test_search_returns_miss_outside_threshold(tmp_path) -> None:
    vector_store = ChromaVectorStore(similarity_threshold=0.8, persist_path=str(tmp_path))
    vector_store.store(
        prompt="What is semantic caching?",
        response="cached response",
        vector=[1.0, 0.0],
    )

    result = vector_store.search_similar([0.0, 1.0])

    assert result.found is False
    assert result.response == ""


def test_chroma_store_persists_entries_between_instances(tmp_path) -> None:
    first_store = ChromaVectorStore(
        similarity_threshold=0.8,
        persist_path=str(tmp_path),
        collection_name="persist_test",
    )
    first_store.store(prompt="prompt", response="response", vector=[1.0, 0.0])

    second_store = ChromaVectorStore(
        similarity_threshold=0.8,
        persist_path=str(tmp_path),
        collection_name="persist_test",
    )

    result = second_store.search_similar([1.0, 0.0])

    assert result.found is True
    assert result.response == "response"


def test_chroma_store_without_eviction_policy_does_not_apply_custom_eviction(tmp_path) -> None:
    vector_store = ChromaVectorStore(persist_path=str(tmp_path))

    assert vector_store.eviction_policy is None


def test_chroma_store_without_eviction_policy_does_not_enforce_max_capacity(tmp_path) -> None:
    vector_store = ChromaVectorStore(
        similarity_threshold=0.1,
        persist_path=str(tmp_path),
        max_capacity=1,
    )
    vector_store.store(prompt="first", response="first response", vector=[0.0, 0.1])
    vector_store.store(prompt="second", response="second response", vector=[10.0, 10.0])

    first_result = vector_store.search_similar([0.0, 0.1])
    second_result = vector_store.search_similar([10.0, 10.0])

    assert first_result.found is True
    assert first_result.response == "first response"
    assert second_result.found is True
    assert second_result.response == "second response"


def test_store_evicts_least_recently_used_entry_when_capacity_is_full(tmp_path) -> None:
    vector_store = ChromaVectorStore(
        similarity_threshold=0.1,
        persist_path=str(tmp_path),
        max_capacity=2,
        eviction_policy=LRUEvictionPolicy(),
    )
    vector_store.store(prompt="first", response="first response", vector=[0.0, 0.1])
    vector_store.store(prompt="second", response="second response", vector=[10.0, 10.0])

    vector_store.search_similar([0.0, 0.1])
    vector_store.store(prompt="third", response="third response", vector=[0.1, 0.0])

    first_result = vector_store.search_similar([0.0, 0.1])
    second_result = vector_store.search_similar([10.0, 10.0])
    third_result = vector_store.search_similar([0.1, 0.0])

    assert first_result.found is True
    assert first_result.response == "first response"
    assert second_result.found is False
    assert third_result.found is True
    assert third_result.response == "third response"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_rejects_invalid_similarity_threshold(tmp_path, threshold: float) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        ChromaVectorStore(similarity_threshold=threshold, persist_path=str(tmp_path))


def test_rejects_invalid_max_capacity(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_capacity"):
        ChromaVectorStore(max_capacity=0, persist_path=str(tmp_path))


def test_rejects_zero_vectors(tmp_path) -> None:
    vector_store = ChromaVectorStore(persist_path=str(tmp_path))

    with pytest.raises(ValueError, match="zero vectors"):
        vector_store.store(prompt="prompt", response="response", vector=[0.0, 0.0])
