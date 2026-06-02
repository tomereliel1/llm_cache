from llm_cache.embedding import OllamaEmbedder
from llm_cache.llm import GroqLLMProvider, OllamaLLMProvider
from llm_cache.test_doubles import VectorStoreMissStub

EMBEDDING_MODEL_NAME = "embeddinggemma"
OLLANA_LLM_MODEL_NAME = "llama3.2:3b"
PROMPT = "What is the capital of Israel?"
EMPTY_PROMPT = ""


def main() -> None:
    print("Welcome to MONOLIT run !!!")
    print("---------------------------------------------------------")

    embedder = OllamaEmbedder(EMBEDDING_MODEL_NAME)
    vector_store = VectorStoreMissStub()
    ollama_llm_provider = OllamaLLMProvider(OLLANA_LLM_MODEL_NAME)
    groq_llm_provider = GroqLLMProvider()
    embedding_vector = embedder.embed(PROMPT)
    print(embedding_vector)

    try:
        embedder.embed(EMPTY_PROMPT)
    except ValueError:
        print("empty prompt throw exception")

    vector_store_res = vector_store.search_similar(embedding_vector)
    if vector_store_res.found:
        print(vector_store_res)
    else:
        print("mock return false as required")
        ollama_answer = ollama_llm_provider.generate_answer(PROMPT)
        print("Ollama's answer")
        print(ollama_answer)
        groq_answer = groq_llm_provider.generate_answer(PROMPT)
        print("Groq's answer")
        print(groq_answer)


if __name__ == "__main__":
    main()
