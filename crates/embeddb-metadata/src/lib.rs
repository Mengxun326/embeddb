//! EmbedDB Metadata Storage
//!
//! Stores document metadata as JSON blobs and provides filtering
//! capabilities. Phase 0 implements basic equality and simple
//! comparison filters. Phase 3 will add inverted indexes for
//! faster filtering on large collections.

pub mod filter;
pub mod store;

mod error;

pub use error::{MetadataError, Result};
