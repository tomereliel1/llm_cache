from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    """Interface for components that generate text answers for prompts."""

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """Generate an answer for a prompt.

        Args:
            prompt: Non-empty user prompt.

        Returns:
            Generated answer text.

        Raises:
            ValueError: If the prompt is invalid for the implementation.
            RuntimeError: If the backing LLM provider cannot complete the request.
        """
