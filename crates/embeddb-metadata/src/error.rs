//! Error types for the metadata module.

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Document not found: id={0}")]
    DocumentNotFound(String),

    #[error("Invalid filter expression: {0}")]
    InvalidFilter(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, MetadataError>;
