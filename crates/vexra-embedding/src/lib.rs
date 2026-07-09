//! Vexra Embedding Engine
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
#[cfg(feature = "onnx")]
pub mod onnx;

pub use simple::SimpleEmbedder;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedder;

/// Trait for text embedding backends.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn dimension(&self) -> usize;
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// L2-normalize a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v { *x /= norm; }
    }
}

/// Create an embedder. Returns an OnnxEmbedder if the `onnx` feature is enabled
/// and the model is available; falls back to SimpleEmbedder otherwise.
pub fn create_embedder(dimension: usize) -> Box<dyn Embedder> {
    #[cfg(feature = "onnx")]
    {
        match onnx::OnnxEmbedder::new(dimension) {
            Ok(e) => return Box::new(e),
            Err(_) => log::warn!("ONNX model not available, falling back to SimpleEmbedder"),
        }
    }
    Box::new(SimpleEmbedder::new(dimension))
}
