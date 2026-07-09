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
use ort::session::Session;
use tokenizers::Tokenizer;

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
    ///
    /// Note: the returned embedder's `dimension()` reflects the **actual model output**,
    /// which may differ from the requested dimension if the parameter was unsupported.
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

        // Load ONNX session and discover actual output dimension
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::OptimizationLevel::Basic)?
            .commit_from_file(model_path)?;

        // Extract actual output dimension from the model metadata
        let actual_dim = Self::infer_dimension(&session)?;

        Ok(Self { session, tokenizer, dimension: actual_dim })
    }

    /// Create from a local model path (skip download).
    /// The caller is responsible for ensuring `dimension` matches the model.
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

    /// Try to infer the output dimension from the ONNX model.
    fn infer_dimension(session: &Session) -> Result<usize, Box<dyn std::error::Error>> {
        for output in &session.outputs {
            if output.name.contains("embed") || output.name.contains("pool") || output.name.contains("last_hidden") {
                if let Some(dim) = output.output_type.tensor_type() {
                    if dim.len() >= 2 {
                        return Ok(dim[dim.len() - 1] as usize);
                    }
                }
            }
        }
        // Fallback: try any output with a known hidden_size
        for output in &session.outputs {
            if let Some(dim) = output.output_type.tensor_type() {
                if dim.len() >= 2 {
                    return Ok(dim[dim.len() - 1] as usize);
                }
            }
        }
        Err("Cannot infer dimension from ONNX model outputs".into())
    }

    /// Mean-pool token embeddings, taking the attention mask into account.
    fn mean_pool(embeddings: &ort::value::Tensor<f32>, attention_mask: &[i64]) -> Vec<f32> {
        let shape = embeddings.shape();
        assert!(shape.len() >= 3, "token_embeddings tensor must have 3+ dimensions");
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let view = embeddings.view();
        let data: Vec<f32> = view.iter().copied().collect();
        let mut result = vec![0.0f32; hidden_size];

        for i in 0..seq_len {
            let mask = attention_mask.get(i).copied().unwrap_or(1) as f32;
            let offset = i * hidden_size;
            for j in 0..hidden_size {
                result[j] += data[offset + j] * mask;
            }
        }
        let mask_sum: f32 = attention_mask.iter().map(|&m| m as f32).sum();
        if mask_sum > 0.0 {
            for v in &mut result { *v /= mask_sum; }
        }
        // L2 normalize
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result { *v /= norm; }
        }
        result
    }
}

impl Embedder for OnnxEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        match self.try_embed(text) {
            Ok(v) => v,
            Err(e) => panic!("ONNX embedding failed: {}", e),
        }
    }
}

impl OnnxEmbedder {
    fn try_embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| format!("tokenize: {}", e))?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = token_ids.len().min(MAX_LENGTH);

        let input_ids = ort::value::Tensor::from_array(([1i64], [seq_len]), &token_ids[..seq_len])?;
        let attn = ort::value::Tensor::from_array(([1i64], [seq_len]), &attention_mask[..seq_len])?;
        let token_type = ort::value::Tensor::from_array(([1i64], [seq_len]), &vec![0i64; seq_len])?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attn,
            "token_type_ids" => token_type,
        ]?)?;

        // Find the embedding tensor by trying known output names
        let embeddings = outputs.iter()
            .find(|(k, _)| k.contains("embed") || k.contains("hidden") || k.contains("pool"))
            .or_else(|| outputs.iter().next())
            .map(|(_, v)| v)
            .ok_or("No output tensor found")?;

        let tensor = embeddings.try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract f32 tensor: {}", e))?;

        Ok(Self::mean_pool(&tensor, &attention_mask[..seq_len]))
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
