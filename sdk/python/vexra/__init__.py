"""Vexra Python SDK — embedded vector database."""

from vexra._native import EmbedDB as Database, PyCollection

Collection = PyCollection
__version__ = "1.1.0"
__all__ = ["Database", "Collection"]
