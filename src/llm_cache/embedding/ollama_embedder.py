import ollama

from llm_cache.health.health_check_result import HealthCheckResult

from .i_embedder import IEmbedder


class OllamaEmbedder(IEmbedder):
    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")

        self._model_name = model_name

    def embed(self, prompt: str) -> list[float]:
        if not prompt:
            raise ValueError("prompt must not be empty")

        response = ollama.embed(
            model=self._model_name,
            input=prompt,
        )

        embeddings = response.get("embeddings")

        if not embeddings:
            raise RuntimeError("Ollama did not return an embedding")

        return embeddings[0]

    def health_check(self) -> HealthCheckResult:
        try:
            embedding = self.embed("health check")

            if not embedding:
                return HealthCheckResult.fail(
                    name="embedding:ollama",
                    message="Ollama returned an empty embedding",
                )

            return HealthCheckResult.ok(
                name="embedding:ollama",
                message=f"Ollama embedding model '{self._model_name}' is working",
            )
        except Exception as error:
            return HealthCheckResult.fail(
                name="embedding:ollama",
                message=f"Ollama embedding model '{self._model_name}' is not working",
                details=str(error),
            )
