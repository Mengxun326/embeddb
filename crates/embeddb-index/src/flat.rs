//! Flat (brute-force) exact nearest neighbor search.
//!
//! Compares the query against every vector in the index. Suitable for
//! collections up to ~50k vectors. For larger collections, HNSW (Phase 1)
//! provides approximate search with much better performance.

use crate::error::{IndexError, Result};
use crate::{DistanceMetric, SearchResult, VectorIndex};

/// A flat index that stores vectors in memory and performs brute-force search.
pub struct FlatIndex {
    /// Stored vectors: (id, vector).
    vectors: Vec<(u64, Vec<f32>)>,
    /// Vector dimension.
    dimension: usize,
    /// Distance metric.
    metric: DistanceMetric,
}

impl FlatIndex {
    /// Create a new empty flat index.
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
            metric,
        }
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get all vectors as (id, vector) pairs.
    pub fn all_vectors(&self) -> &[(u64, Vec<f32>)] {
        &self.vectors
    }

    /// Get a specific vector by internal index.
    pub fn get_by_idx(&self, idx: usize) -> Option<&(u64, Vec<f32>)> {
        self.vectors.get(idx)
    }

    /// Find the internal index for a given vector ID.
    pub fn find_idx(&self, id: u64) -> Option<usize> {
        self.vectors.iter().position(|(vid, _)| *vid == id)
    }

    /// Sort results by score (ascending).
    fn sort_results(results: &mut [SearchResult]) {
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    }
}

impl VectorIndex for FlatIndex {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let k = k.min(self.vectors.len());

        // Compute distance to every vector
        let mut results: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = self.metric.compute(query, vec);
                SearchResult::new(*id, score)
            })
            .collect();

        // Sort ascending (for all our metrics, lower = better)
        Self::sort_results(&mut results);
        results.truncate(k);

        Ok(results)
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        // Replace if ID already exists
        if let Some(idx) = self.find_idx(id) {
            self.vectors[idx] = (id, vector.to_vec());
        } else {
            self.vectors.push((id, vector.to_vec()));
        }

        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        let idx = self
            .find_idx(id)
            .ok_or(IndexError::VectorNotFound(id))?;
        self.vectors.swap_remove(idx);
        Ok(())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_index() -> FlatIndex {
        let mut idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        idx.insert(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, &[0.0, 0.0, 1.0]).unwrap();
        idx
    }

    #[test]
    fn test_search_euclidean() {
        let idx = make_test_index();
        let results = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        // Vector 1 (identical) should be closest
        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 0.0).abs() < 1e-6);
        // Vectors 2 and 3 should be equidistant: sqrt(1^2 + 1^2) = sqrt(2)
        assert!((results[1].score - 2.0f32.sqrt()).abs() < 1e-6);
        assert!((results[2].score - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_search_top_k() {
        let idx = make_test_index();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let idx = make_test_index();
        let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, IndexError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_insert_replace() {
        let mut idx = make_test_index();
        idx.insert(1, &[5.0, 5.0, 5.0]).unwrap();
        assert_eq!(idx.len(), 3);
        let v = idx.get_by_idx(idx.find_idx(1).unwrap()).unwrap();
        assert_eq!(v.1, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_remove() {
        let mut idx = make_test_index();
        idx.remove(2).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx.find_idx(2).is_none());
    }

    #[test]
    fn test_cosine_search() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Cosine);
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 1.0]).unwrap();
        idx.insert(3, &[1.0, 1.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 3).unwrap();
        // Vector 1 is identical (cosine=0), vector 3 is closest (45°, cosine~0.293), vector 2 is orthogonal (cosine=1)
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 3);
        assert_eq!(results[2].id, 2);
    }
}
