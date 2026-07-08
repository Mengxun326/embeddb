//! Error types for the vector index module.

/// Error type for index operations.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector not found: id={0}")]
    VectorNotFound(u64),

    #[error("Index is empty")]
    EmptyIndex,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, IndexError>;
