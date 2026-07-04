from dataclasses import dataclass


@dataclass(frozen=True)
class CacheEntryMetadata:
    id: str
    created_at: float
    last_accessed_at: float
