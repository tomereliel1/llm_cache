from llm_cache.vector_store import IVectorStore, VectorStoreResult


class VectorStoreMissStub(IVectorStore):
    """Vector-store stub that always returns a cache miss."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold
        self.insert_calls_count = 0

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        return VectorStoreResult(found=False, prompt="", response="")

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        self.insert_calls_count += 1
        return "inserted"
