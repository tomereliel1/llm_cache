from abc import ABC, abstractmethod


class IEmbedder(ABC):
    @abstractmethod
    def embed(self, prompt: str) -> list[float]:
        """Return an embedding vector for the given prompt."""
