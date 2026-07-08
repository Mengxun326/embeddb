use std::io;
use thiserror::Error;

/// Specialized error type for storage operations.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid database file: {0}")]
    InvalidFile(String),

    #[error("Page not found: page_id={0}")]
    PageNotFound(u64),

    #[error("WAL corrupted: {0}")]
    WalCorrupted(String),

    #[error("CRC mismatch on page {0}")]
    CrcMismatch(u64),

    #[error("Database is read-only")]
    ReadOnly,

    #[error("Write conflict: another writer is active")]
    WriteConflict,

    #[error("Page is full: page_id={page_id}, needed={needed}, available={available}")]
    PageFull {
        page_id: u64,
        needed: u16,
        available: u16,
    },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u32),

    #[error("{0}")]
    Other(String),
}

/// Convenience result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;
