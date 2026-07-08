//! Configuration types for EmbedDB databases and collections.

use embeddb_index::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Top-level database configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Page size in bytes (default: 4096).
    pub page_size: u32,

    /// Maximum number of pages in the in-memory cache (default: 16384 = 64MB with 4KB pages).
    pub cache_max_pages: usize,

    /// WAL auto-checkpoint threshold (frames).
    pub wal_checkpoint_threshold: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,
            cache_max_pages: 16384,
            wal_checkpoint_threshold: 1000,
        }
    }
}

/// Configuration for creating a new collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name (must be unique within a database).
    pub name: String,

    /// Vector dimension (number of f32 elements per vector).
    pub dimension: usize,

    /// Distance metric for similarity search.
    #[serde(default = "default_distance")]
    pub distance: DistanceMetric,

    /// Optional description.
    #[serde(default)]
    pub description: String,

    /// Page ID for the vector data page (0 = not yet allocated).
    #[serde(default)]
    pub data_root_page: u64,

    /// Page ID for the metadata page (0 = not yet allocated).
    #[serde(default)]
    pub metadata_root_page: u64,
}

fn default_distance() -> DistanceMetric {
    DistanceMetric::Cosine
}

impl CollectionConfig {
    /// Create a new collection config with the given name and dimension.
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            distance: DistanceMetric::Cosine,
            description: String::new(),
            data_root_page: 0,
            metadata_root_page: 0,
        }
    }

    /// Set the distance metric.
    pub fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// A document to insert into a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Optional document ID (auto-generated if not provided).
    pub id: Option<String>,

    /// The vector data (f32 array). Required if no embedding engine is active.
    #[serde(default)]
    pub vector: Option<Vec<f32>>,

    /// Arbitrary JSON metadata.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,

    /// Optional text content (for future embedding engine).
    #[serde(default)]
    pub text: Option<String>,
}

impl Document {
    /// Create a document with a vector.
    pub fn with_vector(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: Some(id.into()),
            vector: Some(vector),
            metadata: None,
            text: None,
        }
    }

    /// Create a document with vector and metadata.
    pub fn with_vector_and_metadata(
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: Some(id.into()),
            vector: Some(vector),
            metadata: Some(metadata),
            text: None,
        }
    }

    /// Create a document with text (for future embedding support).
    pub fn with_text(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            vector: None,
            metadata: None,
            text: Some(text.into()),
        }
    }
}

/// A search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The query vector (f32 array). Required for pure vector search.
    #[serde(default)]
    pub vector: Option<Vec<f32>>,

    /// Optional text query (for future embedding/hybrid search).
    #[serde(default)]
    pub text: Option<String>,

    /// Optional metadata filter expression.
    #[serde(default)]
    pub filter: Option<String>,

    /// Number of results to return (top-k).
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Whether to include vectors in the results.
    #[serde(default)]
    pub include_vectors: bool,

    /// Whether to include metadata in the results.
    #[serde(default = "default_true")]
    pub include_metadata: bool,
}

fn default_top_k() -> usize {
    10
}

fn default_true() -> bool {
    true
}

impl SearchQuery {
    /// Create a vector search query.
    pub fn with_vector(vector: Vec<f32>, top_k: usize) -> Self {
        Self {
            vector: Some(vector),
            text: None,
            filter: None,
            top_k,
            include_vectors: false,
            include_metadata: true,
        }
    }

    /// Set a metadata filter.
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Include vectors in the results.
    pub fn with_vectors(mut self) -> Self {
        self.include_vectors = true;
        self
    }
}

/// A search result (hit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Document ID.
    pub id: String,

    /// Similarity score (lower = more similar for distance metrics).
    pub score: f32,

    /// The vector data (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,

    /// Document metadata (if requested and present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    /// Path to the database file.
    pub path: String,

    /// File size in bytes.
    pub file_size: u64,

    /// Page size in bytes.
    pub page_size: u32,

    /// Total number of pages.
    pub page_count: u64,

    /// Number of collections.
    pub collection_count: usize,

    /// Per-collection statistics.
    pub collections: Vec<CollectionStats>,
}

/// Per-collection statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    /// Collection name.
    pub name: String,

    /// Vector dimension.
    pub dimension: usize,

    /// Distance metric.
    pub distance: String,

    /// Number of vectors.
    pub vector_count: usize,

    /// Number of metadata entries.
    pub metadata_count: usize,
}
