//! ONNX Runtime-based text embedder.
//!
//! Provides production-quality embeddings using transformer models
//! (e.g., all-MiniLM-L6-v2) via ONNX Runtime.
//!
//! # Requirements
//!
//! Enable with: `cargo build --features onnx`
//!
//! The ONNX Runtime C library must be installed:
//! - **Linux**: `apt install libonnxruntime` or download from GitHub
//! - **macOS**: `brew install onnxruntime`
//! - **Windows**: download from https://github.com/microsoft/onnxruntime/releases
//!
//! On first use, the model is automatically downloaded from HuggingFace Hub.

use crate::Embedder;
use hf_hub::api::sync::Api;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use tokenizers::tokenizer::{Result as TokenizerResult, Tokenizer};

/// Default model: all-MiniLM-L6-v2 (384 dimensions, ~80MB, good for English).
const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Maximum token length for the default model.
const MAX_LENGTH: usize = 256;

/// ONNX Runtime-based text embedder.
pub struct OnnxEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder. Downloads the model on first use.
    ///
    /// The `dimension` parameter selects:
    /// - 384 → all-MiniLM-L6-v2 (default, small, fast)
    /// - 768 → all-mpnet-base-v2 (larger, higher quality)
    pub fn new(dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let model_id = match dimension {
            384 => DEFAULT_MODEL,
            768 => "sentence-transformers/all-mpnet-base-v2",
            _ => DEFAULT_MODEL,
        };

        // Download model from HuggingFace Hub
        let api = Api::new()?;
        let model = api.model(model_id.to_string());
        let model_path = model.get("model.onnx")?;

        // Download tokenizer
        let tokenizer_path = api.model(model_id.to_string()).get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer: {}", e))?;

        // Load ONNX session
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::OptimizationLevel::Basic)?
            .commit_from_file(model_path)?;

        Ok(Self { session, tokenizer, dimension })
    }

    /// Create from a local model path (skip download).
    pub fn from_local(
        model_path: &str,
        tokenizer_path: &str,
        dimension: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer: {}", e))?;
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::OptimizationLevel::Basic)?
            .commit_from_file(model_path)?;
        Ok(Self { session, tokenizer, dimension })
    }

    /// Mean-pool the token embeddings, taking the attention mask into account.
    fn mean_pool(output: &SessionOutputs, attention_mask: &[i64]) -> Vec<f32> {
        let embeddings = output["token_embeddings"]
            .try_extract_tensor::<f32>()
            .expect("token_embeddings tensor");

        let shape = embeddings.shape();
        let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);
        let data = embeddings.view().to_vec();
        let mut result = vec![0.0f32; hidden_size];

        for i in 0..seq_len {
            let mask = attention_mask.get(i).copied().unwrap_or(1) as f32;
            for j in 0..hidden_size {
                result[j] += data[i * hidden_size + j] * mask;
            }
        }
        let mask_sum: f32 = attention_mask.iter().map(|&m| m as f32).sum();
        if mask_sum > 0.0 {
            for v in &mut result {
                *v /= mask_sum;
            }
        }
        // L2 normalize
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }
        result
    }
}

impl Embedder for OnnxEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let dim = self.dimension;
        self.try_embed(text).unwrap_or_else(|e| {
            log::warn!("ONNX embedding failed: {}, falling back to zero vector", e);
            vec![0.0; dim]
        })
    }
}

impl OnnxEmbedder {
    fn try_embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| format!("tokenize: {}", e))?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = token_ids.len().min(MAX_LENGTH);

        let input_ids = Tensor::from_array(([1i64], [seq_len]), &token_ids[..seq_len])?;
        let attn = Tensor::from_array(([1i64], [seq_len]), &attention_mask[..seq_len])?;
        let token_type = Tensor::from_array(([1i64], [seq_len]), &vec![0i64; seq_len])?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attn,
            "token_type_ids" => token_type,
        ]?)?;

        Ok(Self::mean_pool(&outputs, &attention_mask[..seq_len]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ONNX Runtime and model download"]
    fn test_onnx_embedder() {
        let embedder = OnnxEmbedder::new(384).unwrap();
        assert_eq!(embedder.dimension(), 384);
        let v = embedder.embed("Hello world");
        assert_eq!(v.len(), 384);
    }
}
