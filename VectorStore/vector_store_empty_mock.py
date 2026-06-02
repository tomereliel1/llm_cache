from .i_vector_store import IVectorStore, VectorStoreResult


class VsEmptyMock(IVectorStore):
    def __init__(self):
        pass

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        return VectorStoreResult(False, "", "")

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        return "inserted"
