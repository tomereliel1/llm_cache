from llm_cache.vector_store import IVectorStore, VectorStoreResult


class VectorStoreHitStub(IVectorStore):
    """Vector-store stub that always returns a cache hit."""

    def __init__(
        self,
        cached_prompt: str = "cached prompt",
        cached_response: str = "cached answer",
    ) -> None:
        self.cached_prompt = cached_prompt
        self.cached_response = cached_response
        self.insert_calls_count = 0

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        return VectorStoreResult(
            found=True,
            prompt=self.cached_prompt,
            response=self.cached_response,
        )

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        self.insert_calls_count += 1
        return "inserted"
