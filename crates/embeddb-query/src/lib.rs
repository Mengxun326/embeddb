//! EmbedDB Query Engine
//!
//! Provides BM25 sparse retrieval and Reciprocal Rank Fusion (RRF)
//! for hybrid search combining dense vectors (HNSW) and sparse text (BM25).
//!
//! # Architecture
//!
//! ```text
//! SearchQuery { text, vector, filter, top_k }
//!     ├── BM25 Search (sparse) → sparse_results
//!     ├── HNSW/Flat Search (dense) → dense_results
//!     └── RRF Fusion → final top_k
//! ```

pub mod bm25;
pub mod fusion;

pub use bm25::Bm25Index;
pub use fusion::FusionStrategy;
