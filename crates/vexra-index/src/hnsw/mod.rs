//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
//!
//! Based on the paper "Efficient and robust approximate nearest neighbor search
//! using Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2016).
//!
//! # Key Concepts
//!
//! HNSW builds a multi-layer graph where each layer is a navigable small-world graph.
//! The bottom layer (layer 0) contains all vectors; higher layers contain progressively
//! fewer vectors (exponentially decaying). Search starts at the top layer's entry point
//! and greedily descends, switching to beam search at lower layers for precision.
//!
//! # Parameters
//!
//! - `M`: Maximum number of connections per node per layer (default: 16)
//! - `M_max`: Maximum connections for layer 0 (default: 2*M)
//! - `ef_construction`: Beam width during index construction (default: 200)
//! - `ef_search`: Beam width during search queries (default: 64)
//!
//! # Persistence
//!
//! The HNSW graph is stored persistently using the page cache:
//! - Each node's layer + vector data is stored in a Vector page cell
//! - Each node's adjacency lists are stored in HNSW Edge page cells
//! - The entry point and global parameters are stored in collection metadata

pub mod builder;
pub mod graph;
pub mod node;
pub mod search;

use serde::{Deserialize, Serialize};

/// Configuration for HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum connections per node per layer (excluding layer 0).
    /// Typical values: 4–64. Default: 16.
    pub m: usize,

    /// Maximum connections for layer 0 nodes (typically 2 * M).
    /// Default: 2 * m.
    pub m_max: usize,

    /// Beam width during index construction. Larger values improve recall
    /// at the cost of slower insertion. Typical: 100–800. Default: 200.
    pub ef_construction: usize,

    /// Beam width during search. Can be tuned per-query. Default: 64.
    pub ef_search: usize,

    /// Normalization factor for level generation.
    /// Default: 1.0 / ln(M).
    #[serde(skip)]
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max: 2 * m,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Create a new config with the given M value (other params auto-tuned).
    pub fn new(m: usize) -> Self {
        Self {
            m,
            m_max: 2 * m,
            ml: 1.0 / (m as f64).ln(),
            ..Default::default()
        }
    }

    /// Set ef_construction.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}
