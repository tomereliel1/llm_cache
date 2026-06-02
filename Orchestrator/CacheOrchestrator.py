from dataclasses import dataclass

from Embedding import IEmbedder
from LLM import ILLMProvider
from VectorStore import IVectorStore


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
        similarity_threshold: float = 0.85,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self._embedder = embedder
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._similarity_threshold = similarity_threshold

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
