# EmbedDB Python SDK

Python bindings for EmbedDB — the embedded vector database.

## Installation

```bash
pip install embeddb
```

## Quick Start

```python
import embeddb

# Open or create a database
db = embeddb.Database("data.embeddb")

# Create a collection
col = db.create_collection("docs", dimension=384, distance=embeddb.DistanceMetric.COSINE)

# Insert vectors
col.insert({"id": "doc1", "vector": [0.1] * 384, "metadata": {"title": "Hello"}})
col.insert({"id": "doc2", "vector": [0.2] * 384, "metadata": {"title": "World"}})

# Search
results = col.search(vector=[0.15] * 384, top_k=10)
for hit in results:
    print(f"{hit.id}: {hit.score:.4f}")
```

## API

### Database

- `Database(path)` — Open or create a database
- `db.create_collection(name, dimension, distance)` — Create a collection
- `db.get_collection(name)` — Get an existing collection
- `db.close()` — Close the database

### Collection

- `col.insert(doc)` — Insert a document `{"id": str, "vector": [float], "metadata": dict}`
- `col.search(vector, top_k=10)` — Search for nearest neighbors

### Distance Metrics

- `DistanceMetric.COSINE` — Cosine distance
- `DistanceMetric.EUCLIDEAN` — Euclidean (L2) distance
- `DistanceMetric.DOT_PRODUCT` — Dot product similarity
