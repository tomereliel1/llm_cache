from dataclasses import dataclass


@dataclass(frozen=True)
class CacheEntryMetadata:
    """Metadata used by eviction policies to choose cache entries.

    Attributes:
        id: Store-specific cache entry identifier.
        created_at: Monotonic timestamp recorded when the entry was created.
        last_accessed_at: Monotonic timestamp recorded when the entry was last used.
    """

    id: str
    created_at: float
    last_accessed_at: float
