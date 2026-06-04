from llm_cache.llm import ILLMProvider


class LLMProviderSpy(ILLMProvider):
    """LLM provider test double that records calls and returns a fixed answer."""

    def __init__(self, answer: str = "generated answer") -> None:
        self.answer = answer
        self.calls_count = 0

    def generate_answer(self, prompt: str) -> str:
        self.calls_count += 1
        return self.answer
