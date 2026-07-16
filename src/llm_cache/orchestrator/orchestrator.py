import logging
from dataclasses import dataclass

from llm_cache.embedding import IEmbedder
from llm_cache.llm import ILLMProvider
from llm_cache.vector_store import IVectorStore

logger = logging.getLogger(__name__)


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
        logger.info("query_started prompt_length=%s", len(clean_prompt))
        embedding_vector = self._embedder.embed(clean_prompt)
        logger.info("embedding_completed vector_size=%s", len(embedding_vector))

        cached_result = self._vector_store.search_similar(embedding_vector)
        if cached_result.found:
            logger.info("cache_hit")
            return QueryResult(
                response=cached_result.response,
                cache_hit=True,
                score=getattr(cached_result, "score", None),
            )

        logger.info("cache_miss")
        generated_response = self._llm_provider.generate_answer(clean_prompt)
        logger.info("llm_completed response_length=%s", len(generated_response))
        self._vector_store.store(
            prompt=clean_prompt,
            response=generated_response,
            vector=embedding_vector,
        )
        logger.info("response_stored")

        return QueryResult(
            response=generated_response,
            cache_hit=False,
        )
