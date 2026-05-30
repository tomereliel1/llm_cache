from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorStoreResult:
    found: bool
    prompt: str
    response: str


class IVectorStore(ABC):
    @abstractmethod
    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        """Return the nearest cached response for the given embedding vector."""

    @abstractmethod
    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        """Store a prompt, its response, and the prompt embedding; return the item id."""
