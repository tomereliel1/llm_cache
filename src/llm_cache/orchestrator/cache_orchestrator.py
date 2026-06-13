from dataclasses import dataclass

from llm_cache.embedding import IEmbedder
from llm_cache.llm import ILLMProvider
from llm_cache.vector_store import IVectorStore


@dataclass(frozen=True)
class QueryResult:
    response: str
    cache_hit: bool
    score: float | None = None


class CacheOrchestrator:
    def __init__(
        self,
        embedder: IEmbedder,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
    ) -> None:
        self._embedder = embedder
        self._llm_provider = llm_provider
        self._vector_store = vector_store

    def query(self, prompt: str) -> QueryResult:
        """Return a cached answer when possible, otherwise generate and store a new answer."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        clean_prompt = prompt.strip()
        embedding_vector = self._embedder.embed(clean_prompt)

        cached_result = self._vector_store.search_similar(embedding_vector)
        if cached_result.found:
            return QueryResult(
                response=cached_result.response,
                cache_hit=True,
                score=getattr(cached_result, "score", None),
            )

        generated_response = self._llm_provider.generate_answer(clean_prompt)
        self._vector_store.store(
            prompt=clean_prompt,
            response=generated_response,
            vector=embedding_vector,
        )

        return QueryResult(
            response=generated_response,
            cache_hit=False,
        )
