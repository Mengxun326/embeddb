"""
EmbedDB Python SDK

EmbedDB is an embedded vector database — SQLite for vectors.
One file, zero configuration, runs anywhere.

Quick start:
    import embeddb

    db = embeddb.Database("data.embeddb")
    col = db.create_collection("docs", dimension=384)
    col.insert({"id": "doc1", "vector": [0.1] * 384})
    results = col.search(vector=[0.2] * 384, top_k=10)
"""

from .client import Database, Collection, SearchResult
from .config import CollectionConfig

__version__ = "0.2.0"
__all__ = ["Database", "Collection", "SearchResult", "CollectionConfig"]
