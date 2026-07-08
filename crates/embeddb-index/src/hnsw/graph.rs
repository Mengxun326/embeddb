//! HNSW graph structure — manages nodes, layers, and the entry point.

use super::node::{random_level, HnswNode};
use super::HnswConfig;
use crate::error::{IndexError, Result};
use crate::{DistanceMetric, SearchResult};
use std::collections::HashMap;

/// The HNSW graph: a multi-layer navigable small-world graph.
///
/// # Thread Safety
///
/// The graph supports concurrent reads (search) and exclusive writes (insert/delete).
/// Use external synchronization (e.g., RwLock) for concurrent access.
pub struct HnswGraph {
    /// All nodes in the graph, keyed by ID.
    nodes: HashMap<u64, HnswNode>,
    /// The entry point node ID for the top layer.
    entry_point: Option<u64>,
    /// The highest layer currently in the graph.
    max_layer: usize,
    /// HNSW configuration.
    config: HnswConfig,
    /// Distance metric.
    metric: DistanceMetric,
    /// Vector dimension.
    dimension: usize,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    pub fn new(dimension: usize, metric: DistanceMetric, config: HnswConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            config,
            metric,
            dimension,
        }
    }

    /// Insert a vector into the graph.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        // If the node already exists, remove and re-insert
        if self.nodes.contains_key(&id) {
            self.remove(id)?;
        }

        // Determine the layer for this node
        let level = random_level(self.config.ml, self.max_layer + 1);
        let vector = vector.to_vec();
        let mut node = HnswNode::new(id, vector.clone(), level);

        if self.entry_point.is_none() {
            // First node in the graph
            self.entry_point = Some(id);
            self.max_layer = level;
            self.nodes.insert(id, node);
            return Ok(());
        }

        // Get the current entry point
        let ep_id = self.entry_point.unwrap();
        let ep_layer = self.max_layer;

        // Phase 1: Navigate from top layer down to level+1 using greedy search
        let mut curr_id = ep_id;
        for lc in ((level + 1)..=ep_layer).rev() {
            let results = self.search_layer(&vector, curr_id, 1, lc)?;
            if let Some(first) = results.first() {
                curr_id = first.id;
            }
        }

        // Phase 2: For layers level down to 0, use beam search to find ef_construction neighbors
        for lc in (0..=level.min(ep_layer)).rev() {
            let candidates = self.search_layer(&vector, curr_id, self.config.ef_construction, lc)?;

            // Select M (or M_max for layer 0) best neighbors using diversity heuristic
            let max_conn = if lc == 0 {
                self.config.m_max
            } else {
                self.config.m
            };
            let selected = self.select_neighbors(&vector, &candidates, max_conn, lc);

            // Add bidirectional connections
            for neighbor_result in &selected {
                if neighbor_result.id == id {
                    continue;
                }
                let neighbor_id = neighbor_result.id;
                // Outgoing: node → neighbor
                node.add_neighbor(lc, neighbor_id, max_conn);
                // Incoming: neighbor → node
                if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                    let max_conn_n = if lc == 0 {
                        self.config.m_max
                    } else {
                        self.config.m
                    };
                    neighbor.add_neighbor(lc, id, max_conn_n);
                }
            }

            // Update current for next layer down
            if !candidates.is_empty() {
                curr_id = candidates[0].id;
            }
        }

        // Update entry point if this node is at a higher layer
        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        self.nodes.insert(id, node);
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let ep_id = match self.entry_point {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };

        let mut curr_id = ep_id;

        // Navigate from top layer down to layer 1 using greedy search
        for lc in (1..=self.max_layer).rev() {
            let results = self.search_layer(query, curr_id, 1, lc)?;
            if let Some(first) = results.first() {
                curr_id = first.id;
            }
        }

        // At layer 0, use beam search with ef_search
        let ef = self.config.ef_search.max(k);
        let results = self.search_layer(query, curr_id, ef, 0)?;

        // Return top-k
        let k = k.min(results.len());
        Ok(results[..k].to_vec())
    }

    /// Remove a node from the graph.
    pub fn remove(&mut self, id: u64) -> Result<()> {
        // Collect all neighbor relationships to update
        let updates: Vec<(usize, Vec<u64>)> = {
            let node = self
                .nodes
                .get(&id)
                .ok_or(IndexError::VectorNotFound(id))?;

            (0..=node.max_layer)
                .map(|layer| (layer, node.neighbors_at(layer).to_vec()))
                .collect()
        };

        // Remove connections from neighbors back to this node
        for (layer, neighbor_ids) in updates {
            for neighbor_id in neighbor_ids {
                if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                    neighbor.remove_neighbor(layer, id);
                }
            }
        }

        // Remove the node itself
        self.nodes.remove(&id);

        // If this was the entry point, find a new one
        if self.entry_point == Some(id) {
            self.entry_point = self.nodes.keys().next().copied();
            if self.entry_point.is_none() {
                self.max_layer = 0;
            } else {
                // Recalculate max_layer
                self.max_layer = self
                    .nodes
                    .values()
                    .map(|n| n.max_layer)
                    .max()
                    .unwrap_or(0);
            }
        }

        Ok(())
    }

    /// Mark a node as deleted without removing it (preserves graph connectivity).
    pub fn soft_delete(&mut self, id: u64) -> Result<()> {
        let node = self
            .nodes
            .get_mut(&id)
            .ok_or(IndexError::VectorNotFound(id))?;
        node.mark_deleted();
        Ok(())
    }

    /// Return the number of active (non-tombstoned) nodes.
    pub fn len(&self) -> usize {
        self.nodes.values().filter(|n| n.is_active()).count()
    }

    /// Return true if the graph has no active nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the entry point ID and layer.
    pub fn entry_point(&self) -> Option<(u64, usize)> {
        self.entry_point.map(|id| (id, self.max_layer))
    }

    /// Get a reference to a node by ID.
    pub fn get_node(&self, id: u64) -> Option<&HnswNode> {
        self.nodes.get(&id)
    }

    /// Get the total number of nodes (including tombstoned).
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the graph configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    // ------------------------------------------------------------------
    // Internal search helpers
    // ------------------------------------------------------------------

    /// Search within a single layer using beam search.
    ///
    /// Returns up to `ef` nearest candidates sorted by distance.
    fn search_layer(
        &self,
        query: &[f32],
        entry_id: u64,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<SearchResult>> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        // Min-heap for candidates (we want the closest, so negate score)
        // BinaryHeap is a max-heap by default
        #[derive(PartialEq)]
        struct Candidate {
            id: u64,
            score: f32,
        }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.score.partial_cmp(&self.score)
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut visited: HashMap<u64, f32> = HashMap::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Start from entry point
        let entry = self
            .nodes
            .get(&entry_id)
            .ok_or(IndexError::VectorNotFound(entry_id))?;
        let dist = self.metric.compute(query, &entry.vector);
        visited.insert(entry_id, dist);
        candidates.push(Candidate { id: entry_id, score: dist });
        results.push(Candidate { id: entry_id, score: dist });

        while !candidates.is_empty() {
            let curr = candidates.pop().unwrap();

            // If worst result is better than current candidate, we're done
            if let Some(worst) = results.peek() {
                if results.len() >= ef && curr.score > worst.score {
                    break;
                }
            }

            // Explore neighbors
            if let Some(node) = self.nodes.get(&curr.id) {
                for &neighbor_id in node.neighbors_at(layer) {
                    if neighbor_id == curr.id {
                        continue;
                    }

                    let d = if let Some(&prev_d) = visited.get(&neighbor_id) {
                        prev_d
                    } else {
                        if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                            if neighbor.tombstone {
                                continue;
                            }
                            let d = self.metric.compute(query, &neighbor.vector);
                            visited.insert(neighbor_id, d);
                            d
                        } else {
                            continue;
                        }
                    };

                    // If this neighbor is closer than the worst result, add it
                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(worst) = results.peek() {
                        d < worst.score
                    } else {
                        false
                    };

                    if should_add {
                        candidates.push(Candidate { id: neighbor_id, score: d });
                        results.push(Candidate { id: neighbor_id, score: d });
                        // Prune results to ef
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted SearchResult vec
        let mut sorted: Vec<SearchResult> = results
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchResult::new(c.id, c.score))
            .collect();
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted)
    }

    /// Select best neighbors using the diversity heuristic (Algorithm 4 from the paper).
    ///
    /// From the candidate set, select up to `max_conn` neighbors that are both
    /// close to the query AND diverse (not too close to each other).
    fn select_neighbors(
        &self,
        _query: &[f32],
        candidates: &[SearchResult],
        max_conn: usize,
        _layer: usize,
    ) -> Vec<SearchResult> {
        if candidates.len() <= max_conn {
            return candidates.to_vec();
        }

        // Simple heuristic: take the closest candidates first,
        // then prune those that are closer to already-selected neighbors
        // than they are to the query.
        let mut selected: Vec<SearchResult> = Vec::new();

        for candidate in candidates {
            if selected.len() >= max_conn {
                break;
            }

            // Check diversity: if this candidate is closer to any already-selected
            // node than it is to the query, skip it
            let mut is_diverse = true;
            if let Some(cand_node) = self.nodes.get(&candidate.id) {
                let dist_to_query = candidate.score;

                for sel in &selected {
                    if let Some(sel_node) = self.nodes.get(&sel.id) {
                        let dist_between = self.metric.compute(
                            &cand_node.vector,
                            &sel_node.vector,
                        );
                        if dist_between < dist_to_query {
                            is_diverse = false;
                            break;
                        }
                    }
                }
            }

            if is_diverse {
                selected.push(candidate.clone());
            }
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> HnswConfig {
        HnswConfig::new(8).with_ef_construction(50).with_ef_search(20)
    }

    #[test]
    fn test_insert_and_search() {
        let mut graph = HnswGraph::new(3, DistanceMetric::Euclidean, make_config());

        // Insert vectors
        for i in 0..10u64 {
            let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            graph.insert(i, &v).unwrap();
        }

        assert_eq!(graph.len(), 10);

        // Search
        let query = vec![5.0, 10.0, 15.0];
        let results = graph.search(&query, 3).unwrap();
        assert_eq!(results.len(), 3);
        // The closest should be id=5 (vector [5,10,15])
        assert_eq!(results[0].id, 5);
    }

    #[test]
    fn test_remove() {
        let mut graph = HnswGraph::new(2, DistanceMetric::Cosine, make_config());

        graph.insert(1, &[1.0, 0.0]).unwrap();
        graph.insert(2, &[0.0, 1.0]).unwrap();
        assert_eq!(graph.len(), 2);

        graph.remove(1).unwrap();
        assert_eq!(graph.len(), 1);

        // Should still be able to search
        let results = graph.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_empty_search() {
        let graph = HnswGraph::new(3, DistanceMetric::Euclidean, make_config());
        let results = graph.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut graph = HnswGraph::new(3, DistanceMetric::Euclidean, make_config());
        let err = graph.insert(1, &[1.0, 0.0]).unwrap_err();
        assert!(matches!(err, IndexError::DimensionMismatch { .. }));
    }
}
