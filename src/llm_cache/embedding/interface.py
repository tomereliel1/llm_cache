from abc import ABC, abstractmethod


class IEmbedder(ABC):
    """Interface for components that convert prompts into embedding vectors."""

    @abstractmethod
    def embed(self, prompt: str) -> list[float]:
        """Create an embedding vector for a prompt.

        Args:
            prompt: Non-empty text to embed.

        Returns:
            Numeric embedding vector representing the prompt.

        Raises:
            ValueError: If the prompt is invalid for the implementation.
            RuntimeError: If the backing embedding provider cannot complete the request.
        """
