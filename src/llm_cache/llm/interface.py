from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """Generate an LLM response for the given prompt."""
