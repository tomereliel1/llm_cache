from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an LLM response for the given prompt."""
