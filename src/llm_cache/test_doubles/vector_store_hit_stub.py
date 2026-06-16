from llm_cache.health.health_check_result import HealthCheckResult
from llm_cache.vector_store import IVectorStore, VectorStoreResult


class VectorStoreHitStub(IVectorStore):
    """Vector-store stub that always returns a cache hit."""

    def __init__(
        self,
        cached_prompt: str = "cached prompt",
        cached_response: str = "cached answer",
        similarity_threshold: float = 0.8,
        max_capacity: int = 1000,
    ) -> None:
        self.cached_prompt = cached_prompt
        self.cached_response = cached_response
        self.similarity_threshold = similarity_threshold
        self.max_capacity = max_capacity
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

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult.ok(
            name="vector-store:hit-stub",
            message="VectorStoreHitStub is ready",
        )
