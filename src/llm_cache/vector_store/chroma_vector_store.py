from __future__ import annotations

from time import monotonic
from uuid import uuid4

import chromadb

from llm_cache.vector_store.cache_entry_metadata import CacheEntryMetadata
from llm_cache.vector_store.i_eviction_policy import IEvictionPolicy
from llm_cache.vector_store.i_vector_store import IVectorStore, VectorStoreResult


class ChromaVectorStore(IVectorStore):
    """Persistent vector store backed by Chroma's default distance behavior."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        persist_path: str = ".cache/chroma",
        collection_name: str = "llm_cache",
        max_capacity: int = 1000,
        eviction_policy: IEvictionPolicy | None = None,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if max_capacity < 1:
            raise ValueError("max_capacity must be at least 1")

        if not persist_path.strip():
            raise ValueError("persist_path must be a non-empty string")

        if not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

        self.similarity_threshold = similarity_threshold
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.max_capacity = max_capacity
        self.eviction_policy = eviction_policy

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    def search_similar(self, vector: list[float]) -> VectorStoreResult:
        self._validate_vector(vector)

        if self._collection.count() == 0:
            return VectorStoreResult(found=False, prompt="", response="")

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids") or [[]]
        if not ids[0]:
            return VectorStoreResult(found=False, prompt="", response="")

        entry_id = ids[0][0]
        distance = results["distances"][0][0]
        if distance > self.similarity_threshold:
            return VectorStoreResult(found=False, prompt="", response="")

        document = results["documents"][0][0]
        metadata = results["metadatas"][0][0] or {}
        response = metadata.get("response", "")
        self._touch(entry_id, metadata)

        return VectorStoreResult(
            found=True,
            prompt=document,
            response=str(response),
        )

    def store(self, prompt: str, response: str, vector: list[float]) -> str:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be a non-empty string")

        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")

        self._validate_vector(vector)
        self._evict_if_needed()

        entry_id = str(uuid4())
        now = monotonic()
        self._collection.add(
            ids=[entry_id],
            embeddings=[vector],
            documents=[clean_prompt],
            metadatas=[
                {
                    "response": response,
                    "created_at": now,
                    "last_accessed_at": now,
                }
            ],
        )
        return entry_id

    def _touch(self, entry_id: str, metadata: dict) -> None:
        updated_metadata = {
            **metadata,
            "last_accessed_at": monotonic(),
        }
        self._collection.update(ids=[entry_id], metadatas=[updated_metadata])

    def _evict_if_needed(self) -> None:
        if self.eviction_policy is None:
            return

        if self._collection.count() < self.max_capacity:
            return

        entries = self._collection.get(include=["metadatas"])
        ids = entries.get("ids") or []
        metadatas = entries.get("metadatas") or []
        if not ids:
            return

        victim_id = self.eviction_policy.choose_victim(
            [
                CacheEntryMetadata(
                    id=entry_id,
                    created_at=(metadata or {}).get("created_at", 0),
                    last_accessed_at=(metadata or {}).get("last_accessed_at", 0),
                )
                for entry_id, metadata in zip(ids, metadatas, strict=True)
            ]
        )
        self._collection.delete(ids=[victim_id])

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")

        if all(value == 0 for value in vector):
            raise ValueError("vectors must not be zero vectors")
