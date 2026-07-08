//! EmbedDB Core — Public API for the embedded vector database.
//!
//! This crate provides the main entry point for using EmbedDB. It
//! orchestrates the storage engine, vector indexes, and metadata
//! stores behind a clean, ergonomic API.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use vexra_core::{Database, CollectionConfig, Document, SearchQuery};
//!
//! let db = Database::open("data.embeddb").unwrap();
//! db.create_collection(CollectionConfig::new("docs", 384)).unwrap();
//!
//! let col = db.get_collection("docs").unwrap();
//! let mut col = col.write();
//! col.insert(Document::with_vector("doc1", vec![0.1; 384])).unwrap();
//!
//! let results = col.search(
//!     SearchQuery::with_vector(vec![0.2; 384], 10)
//! ).unwrap();
//! ```

pub mod collection;
pub mod config;
pub mod db;
pub mod error;

// Re-export main types for convenience
pub use collection::{Collection, IndexBackend, IndexType};
pub use config::{
    CollectionConfig, CollectionStats, DatabaseConfig, DatabaseStats, Document, SearchHit,
    SearchQuery,
};
pub use db::{insert, search, Database};
pub use error::{Error, Result};

// Re-export sub-crate types that are part of the public API
pub use vexra_index::DistanceMetric;
