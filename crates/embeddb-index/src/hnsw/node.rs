//! HNSW graph node representation and serialization.

use serde::{Deserialize, Serialize};

/// A node in the HNSW graph.
///
/// Each node stores:
/// - Its vector data (f32 array)
/// - Layer assignment (0 = bottom, higher = sparser)
/// - Adjacency lists per layer (neighbor node IDs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Unique node identifier (maps to vector ID).
    pub id: u64,
    /// The vector data.
    pub vector: Vec<f32>,
    /// The highest layer this node belongs to (0-indexed).
    /// Layer 0 is the bottom (most dense) layer.
    pub max_layer: usize,
    /// Adjacency lists: one Vec<u64> per layer [layer_0_neighbors, layer_1_neighbors, ...].
    pub neighbors: Vec<Vec<u64>>,
    /// Whether this node is tombstoned (soft-deleted).
    pub tombstone: bool,
}

impl HnswNode {
    /// Create a new node with a random layer assignment.
    pub fn new(id: u64, vector: Vec<f32>, level: usize) -> Self {
        let neighbors = vec![Vec::new(); level + 1];
        Self {
            id,
            vector,
            max_layer: level,
            neighbors,
            tombstone: false,
        }
    }

    /// Get the neighbor list for a specific layer.
    pub fn neighbors_at(&self, layer: usize) -> &[u64] {
        if layer < self.neighbors.len() {
            &self.neighbors[layer]
        } else {
            &[]
        }
    }

    /// Add a neighbor at a specific layer (if not already present and within M_max).
    pub fn add_neighbor(&mut self, layer: usize, neighbor_id: u64, max_connections: usize) -> bool {
        if layer >= self.neighbors.len() {
            return false;
        }
        let list = &mut self.neighbors[layer];
        if list.contains(&neighbor_id) {
            return false;
        }
        if list.len() >= max_connections {
            return false;
        }
        list.push(neighbor_id);
        true
    }

    /// Remove a neighbor at a specific layer.
    pub fn remove_neighbor(&mut self, layer: usize, neighbor_id: u64) {
        if layer < self.neighbors.len() {
            self.neighbors[layer].retain(|&id| id != neighbor_id);
        }
    }

    /// Mark this node as deleted (tombstone).
    pub fn mark_deleted(&mut self) {
        self.tombstone = true;
    }

    /// Check if this node is active (not tombstoned).
    pub fn is_active(&self) -> bool {
        !self.tombstone
    }
}

/// Generate a random layer for a new node using the HNSW level generation algorithm.
///
/// The level is floor(-ln(uniform(0,1)) * mL), clamped to a reasonable maximum.
pub fn random_level(ml: f64, max_level: usize) -> usize {
    let r: f64 = fastrand::f64();
    let level = (-r.ln() * ml).floor() as usize;
    level.min(max_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = HnswNode::new(1, vec![1.0, 2.0, 3.0], 2);
        assert_eq!(node.id, 1);
        assert_eq!(node.max_layer, 2);
        assert_eq!(node.neighbors.len(), 3); // layers 0,1,2
        assert!(!node.tombstone);
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = HnswNode::new(1, vec![1.0; 10], 0);
        assert!(node.add_neighbor(0, 2, 32));
        assert_eq!(node.neighbors_at(0), &[2]);
        // Duplicate
        assert!(!node.add_neighbor(0, 2, 32));
        assert_eq!(node.neighbors_at(0).len(), 1);
    }

    #[test]
    fn test_random_level() {
        let ml = 1.0 / (16.0_f64).ln();
        for _ in 0..100 {
            let level = random_level(ml, 10);
            assert!(level <= 10);
        }
    }
}
