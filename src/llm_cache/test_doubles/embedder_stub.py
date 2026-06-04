from llm_cache.embedding import IEmbedder


class EmbedderStub(IEmbedder):
    """Embedder test double that returns a fixed vector."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]

    def embed(self, prompt: str) -> list[float]:
        return self.vector
