import ollama

from .i_llm_provider import ILLMProvider


class OllamaLLMProvider(ILLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answer(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        print("Prompt received. Generating answer, this may take a while...")

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
        )

        return response["response"]
