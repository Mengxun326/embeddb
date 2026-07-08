"""
EmbedDB Python SDK — embedded vector database.

Quick start:
    import embeddb
    db = embeddb.Database("data.embeddb")
    col = db.create_collection("docs", dimension=384)
    col.insert({"vector": [0.1] * 384, "id": "doc1"})
    results = col.search(vector=[0.2] * 384, top_k=10)

The native PyO3 module is loaded from `_native` (compiled by maturin).
"""

from embeddb._native import EmbedDB as Database, PyCollection

# For backward compatibility: expose Collection as an alias
Collection = PyCollection

__version__ = "1.1.0"
__all__ = ["Database", "Collection", "PyCollection"]
