//! Simple hash-based text embedder.
//!
//! Uses a deterministic hash function to project token n-grams
//! into a fixed-dimension vector space. No external dependencies.
//!
//! Quality is lower than transformer-based embeddings but sufficient
//! for testing, prototyping, and basic keyword matching.

use crate::Embedder;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A simple embedder that uses character n-gram hashing + normalization.
pub struct SimpleEmbedder {
    dimension: usize,
}

impl SimpleEmbedder {
    /// Create a new embedder with the given output dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Embedder for SimpleEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dimension];
        let text_lower = text.to_lowercase();

        // Unigram tokens
        for word in text_lower.split_whitespace() {
            let h = hash_str(word);
            let idx = (h % self.dimension as u64) as usize;
            vector[idx] += 1.0;
        }

        // Bigrams (character-level)
        let chars: Vec<char> = text_lower.chars().collect();
        for window in chars.windows(2) {
            let bigram: String = window.iter().collect();
            let h = hash_str(&bigram);
            let idx = (h % self.dimension as u64) as usize;
            vector[idx] += 0.5;
        }

        // Trigams
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let h = hash_str(&trigram);
            let idx = (h % self.dimension as u64) as usize;
            vector[idx] += 0.25;
        }

        // L2 normalize
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        vector
    }
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let embedder = SimpleEmbedder::new(128);
        let v = embedder.embed("hello world");
        assert_eq!(v.len(), 128);
    }

    #[test]
    fn test_similar_texts_are_closer() {
        let embedder = SimpleEmbedder::new(64);
        let a = embedder.embed("vector database");
        let b = embedder.embed("vector search");
        let c = embedder.embed("apple banana");

        // a and b share "vector" → should be more similar (lower distance)
        let ab_sum: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        let ac_sum: f32 = a.iter().zip(&c).map(|(x, y)| (x - y).abs()).sum();
        assert!(ab_sum < ac_sum, "Similar texts should have lower distance");
    }

    #[test]
    fn test_deterministic() {
        let embedder = SimpleEmbedder::new(64);
        let v1 = embedder.embed("test text");
        let v2 = embedder.embed("test text");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_batch() {
        let embedder = SimpleEmbedder::new(32);
        let vs = embedder.embed_batch(&["a", "b", "c"]);
        assert_eq!(vs.len(), 3);
        assert_eq!(vs[0].len(), 32);
    }
}
