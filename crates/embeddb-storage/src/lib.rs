//! EmbedDB Storage Engine
//!
//! Low-level storage primitives for the EmbedDB embedded vector database.
//! Implements a SQLite-inspired page-based file format with Write-Ahead Log
//! for crash safety and concurrent reads via snapshot isolation.
//!
//! # Architecture
//!
//! The storage engine is organized into four main modules:
//!
//! - [`format`] — On-disk page layout and type definitions
//! - [`wal`] — Write-Ahead Log for crash-safe writes
//! - [`page_cache`] — LRU page cache with memory-mapped I/O
//! - [`encoding`] — Binary encoding/decoding helpers for page data

pub mod encoding;
pub mod format;
pub mod page_cache;
pub mod wal;

mod error;

pub use error::{Result, StorageError};

/// Magic bytes that identify an EmbedDB database file.
pub const MAGIC: &[u8; 11] = b"EmbedDB v1\0";

/// Current version of the on-disk format.
pub const FORMAT_VERSION: u32 = 1;

/// Default page size in bytes.
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// Maximum number of pages a single database file can contain.
pub const MAX_PAGES: u64 = 1 << 32; // ~17.6 TB with 4KB pages
