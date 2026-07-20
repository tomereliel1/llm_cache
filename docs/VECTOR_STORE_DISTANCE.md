# Vector Store Distance Functions

The vector store compares the embedding for a new prompt with embeddings already in
the cache. The configured `similarity_threshold` decides whether the nearest cached
entry is close enough to reuse.

## In-Memory Vector Store

The in-memory provider always uses cosine similarity.

Cosine similarity measures the angle between two vectors. Higher is better:

```text
1.0   same direction
0.0   unrelated direction
-1.0  opposite direction
```

The in-memory provider returns a cache hit when:

```text
similarity >= similarity_threshold
```

Example: with `similarity_threshold = 0.8`, a cached entry with cosine similarity
`0.9` is a hit, while `0.7` is a miss.

The in-memory provider ignores Chroma-only distance-function settings.

## Chroma Vector Store

Chroma uses distance functions. Lower is better, and the nearest cached entry is a
hit when:

```text
distance <= similarity_threshold
```

The setting is still named `similarity_threshold` for consistency with the rest of
the application, but for Chroma it acts as the maximum accepted distance.

Supported Chroma distance functions:

- `l2`: squared Euclidean distance. It sums the squared difference between vector
  dimensions. Identical vectors have distance `0`.
- `cosine`: cosine distance, effectively `1 - cosine_similarity`. Identical
  directions have distance `0`; unrelated directions are farther away.
- `ip`: inner-product distance. Chroma ranks vectors by inner product using a
  distance form where lower values are better.

Example: with Chroma `cosine` and `similarity_threshold = 0.2`, a cached entry with
distance `0.1` is a hit, while distance `0.3` is a miss.

## Configuring Chroma

Local one-process CLI:

```bash
uv run python -m demos.local_one_prompt_demo \
  --vector-store-provider chroma \
  --vector-store-distance-function cosine
```

Vector-store gRPC server:

```bash
uv run python -m llm_cache.vector_store.grpc.server \
  --provider chroma \
  --distance-function cosine
```

Runtime configuration:

```json
{
  "vector_store_service": {
    "provider": "chroma",
    "path": "/data/chroma",
    "collection": "llm_cache",
    "distance_function": "cosine",
    "similarity_threshold": 0.8
  }
}
```

## Existing Collections

Chroma fixes the distance function when a collection is created. An existing
collection cannot be changed from `l2` to `cosine`, from `cosine` to `ip`, or any
other combination in place.

If the configured distance function does not match the existing collection, startup
fails with a message naming the collection, the existing function, and the requested
function.

To change distance function, choose one of these options:

- Keep using the existing distance function in configuration.
- Use a new collection name or Chroma path.
- Delete the existing Chroma data so the collection can be created again.
