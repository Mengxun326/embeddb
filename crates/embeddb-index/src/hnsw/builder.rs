//! HNSW index builder — incremental construction with batch insertion support.

use super::graph::HnswGraph;
use super::HnswConfig;
use crate::error::Result;
use crate::DistanceMetric;

/// Builder for constructing an HNSW index incrementally.
pub struct HnswBuilder {
    graph: HnswGraph,
}

impl HnswBuilder {
    /// Create a new builder for an HNSW index.
    pub fn new(dimension: usize, metric: DistanceMetric, config: HnswConfig) -> Self {
        Self {
            graph: HnswGraph::new(dimension, metric, config),
        }
    }

    /// Insert a single vector.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        self.graph.insert(id, vector)
    }

    /// Insert a batch of vectors. Currently sequential; Phase 2 will add parallelism.
    pub fn insert_batch(&mut self, items: &[(u64, Vec<f32>)]) -> Result<()> {
        for (id, vector) in items {
            self.graph.insert(*id, vector)?;
        }
        Ok(())
    }

    /// Remove a vector.
    pub fn remove(&mut self, id: u64) -> Result<()> {
        self.graph.remove(id)
    }

    /// Soft-delete a vector (preserves graph connectivity).
    pub fn soft_delete(&mut self, id: u64) -> Result<()> {
        self.graph.soft_delete(id)
    }

    /// Get the current number of active nodes.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Consume the builder and return the HNSW graph.
    pub fn build(self) -> HnswGraph {
        self.graph
    }

    /// Get a reference to the underlying graph.
    pub fn graph(&self) -> &HnswGraph {
        &self.graph
    }

    /// Get a mutable reference to the underlying graph.
    pub fn graph_mut(&mut self) -> &mut HnswGraph {
        &mut self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_insert() {
        let mut builder = HnswBuilder::new(
            3,
            DistanceMetric::Euclidean,
            HnswConfig::default(),
        );

        let items: Vec<(u64, Vec<f32>)> = (0..50u64)
            .map(|i| (i, vec![i as f32, (i * 2) as f32, (i * 3) as f32]))
            .collect();

        builder.insert_batch(&items).unwrap();
        assert_eq!(builder.len(), 50);

        let graph = builder.build();
        let results = graph.search(&[10.0, 20.0, 30.0], 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 10); // exact match
    }
}
