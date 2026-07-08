//! EmbedDB Vector Index — Flat (brute-force) exact nearest neighbor search.
//!
//! Phase 0 implements exact search by comparing against every vector in the
//! collection. Phase 1 will add HNSW approximate search.
//!
//! # Distance Metrics
//!
//! - Cosine distance: 1 - (a·b)/(||a||·||b||)
//! - Euclidean distance: ||a - b||₂
//! - Dot product similarity: a·b (higher = more similar)

pub mod distance;
pub mod flat;
pub mod hnsw;

mod error;

pub use error::{IndexError, Result};

/// Trait for a vector index that supports search operations.
pub trait VectorIndex {
    /// Search for the k nearest neighbors to the query vector.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Insert a vector with its ID into the index.
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()>;

    /// Remove a vector from the index.
    fn remove(&mut self, id: u64) -> Result<()>;

    /// Return the number of vectors in the index.
    fn len(&self) -> usize;

    /// Return true if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A single search result: a vector ID and its distance/similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The ID of the matching vector.
    pub id: u64,
    /// Distance or similarity score (lower = closer for distance metrics).
    pub score: f32,
}

impl SearchResult {
    pub fn new(id: u64, score: f32) -> Self {
        Self { id, score }
    }
}

/// Supported distance metrics for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity. Range [0, 2].
    Cosine,
    /// Euclidean (L2) distance. Range [0, ∞).
    Euclidean,
    /// Dot product similarity (higher = more similar). Range (-∞, ∞).
    /// Note: For search, we negate to sort ascending.
    DotProduct,
}

impl DistanceMetric {
    /// Compute the distance/similarity between two vectors.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => distance::cosine(a, b),
            DistanceMetric::Euclidean => distance::euclidean(a, b),
            DistanceMetric::DotProduct => -distance::dot_product(a, b), // negate for ascending sort
        }
    }

    /// Returns true if lower scores are better (distance metrics).
    /// Returns false if higher scores are better (similarity metrics).
    pub fn lower_is_better(&self) -> bool {
        match self {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => true,
            DistanceMetric::DotProduct => true, // we negate dot product
        }
    }
}
