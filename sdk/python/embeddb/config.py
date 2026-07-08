"""Configuration types for EmbedDB collections."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistanceMetric(str, Enum):
    """Distance metric for vector similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"


@dataclass
class CollectionConfig:
    """Configuration for creating a new collection."""
    name: str
    dimension: int
    distance: DistanceMetric = DistanceMetric.COSINE
    description: str = ""


@dataclass
class SearchResult:
    """A single search result (hit)."""
    id: str
    score: float
    metadata: Optional[dict] = None
    vector: Optional[list] = None
