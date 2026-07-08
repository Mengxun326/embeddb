//! EmbedDB Embedding Engine
//!
//! Provides text-to-vector embedding for automatic vector generation.
//! Two backends are available:
//!
//! 1. **SimpleEmbedder** (default): Hash-based projection, no external dependencies.
//!    Fast and deterministic, suitable for testing and lightweight use cases.
//! 2. **OnnxEmbedder** (feature = "onnx"): ONNX Runtime based, production-quality
//!    embeddings using models like all-MiniLM-L6-v2 (384d).
//!
//! # Example
//!
//! ```rust,ignore
//! use vexra_embedding::SimpleEmbedder;
//! let embedder = SimpleEmbedder::new(384);
//! let vector = embedder.embed("Hello world");
//! assert_eq!(vector.len(), 384);
//! ```

pub mod simple;
// pub mod onnx;  // Feature-gated behind "onnx"

pub use simple::SimpleEmbedder;

/// Trait for text embedding backends.
pub trait Embedder: Send + Sync {
    /// Embed text into a fixed-dimension vector.
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Get the output dimension.
    fn dimension(&self) -> usize;
    /// Embed multiple texts in a batch.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
