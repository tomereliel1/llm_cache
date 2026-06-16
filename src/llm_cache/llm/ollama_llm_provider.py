import ollama

from llm_cache.health.health_check_result import HealthCheckResult

from .i_llm_provider import ILLMProvider


class OllamaLLMProvider(ILLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answer(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        print("Prompt received. Generating answer, this may take a while...")

        return self._generate_answer(prompt)

    def _generate_answer(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
        )

        return response["response"]

    def health_check(self) -> HealthCheckResult:
        try:
            response = self._generate_answer("Reply with exactly: OK")

            if not response:
                return HealthCheckResult.fail(
                    name="llm:ollama",
                    message="Ollama returned an empty response",
                )

            return HealthCheckResult.ok(
                name="llm:ollama",
                message=f"Ollama LLM model '{self.model_name}' is working",
            )
        except Exception as error:
            return HealthCheckResult.fail(
                name="llm:ollama",
                message=f"Ollama LLM model '{self.model_name}' is not working",
                details=str(error),
            )
