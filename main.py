from Embedding import OllamaEmbedder
from LLM import OllamaLLMProvider
from Orchestrator import CacheOrchestrator
from VectorStore import VsEmptyMock

EMBEDDING_MODEL_NAME = "embeddinggemma"
LLM_MODEL_NAME = "llama3.2:3b"
PROMPT = "What is the capital of Israel?"
SIMILARITY_THRESHOLD = 0.85


def main() -> None:
    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    llm_provider = OllamaLLMProvider(LLM_MODEL_NAME)
    vector_store = VsEmptyMock()

    orchestrator = CacheOrchestrator(
        embedder=embedder,
        llm_provider=llm_provider,
        vector_store=vector_store,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    result = orchestrator.query(PROMPT)

    print(result.response)
    print(f"cache_hit={result.cache_hit}")


if __name__ == "__main__":
    main()
