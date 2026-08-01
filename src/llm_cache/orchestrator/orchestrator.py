import logging
from dataclasses import dataclass

from llm_cache.embedding import IEmbedder
from llm_cache.llm import ILLMProvider
from llm_cache.vector_store import IVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    """Result returned by a cache query.

    Attributes:
        response: Answer returned to the caller.
        cache_hit: Whether the response came from a cached entry.
        score: Optional similarity score reported by the vector store for cache hits.
    """

    response: str
    cache_hit: bool
    score: float | None = None


class CacheOrchestrator:
    """Coordinate embedding, semantic lookup, LLM generation, and cache storage.

    The orchestrator depends only on provider interfaces, so callers may inject local
    providers, test doubles, or gRPC client adapters.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
    ) -> None:
        """Create an orchestrator from provider implementations.

        Args:
            embedder: Component that converts prompts to embedding vectors.
            llm_provider: Component that generates answers on cache misses.
            vector_store: Component that searches and stores cached prompt responses.
        """
        self._embedder = embedder
        self._llm_provider = llm_provider
        self._vector_store = vector_store

    def query(self, prompt: str) -> QueryResult:
        """Answer a prompt using the semantic cache when possible.

        Args:
            prompt: User prompt to answer. Surrounding whitespace is ignored.

        Returns:
            QueryResult describing the answer and whether it came from the cache.

        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
            RuntimeError: If an injected provider reports a runtime failure.
        """
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
