//! Result fusion strategies for hybrid search.
//!
//! Combines dense (HNSW) and sparse (BM25) results into a single ranking.

/// Fusion strategy for combining dense and sparse search results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
    /// k is typically 60 (default below).
    ReciprocalRankFusion,
    /// Weighted linear combination: score = α × dense + (1-α) × sparse
    Weighted { alpha: f32 },
}

/// A fused search result.
#[derive(Debug, Clone)]
pub struct FusedHit {
    pub doc_id: String,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
}

/// Fuse dense and sparse results using RRF.
///
/// RRF (Reciprocal Rank Fusion) combines rankings without requiring
/// score normalization. It works well when dense and sparse scores
/// are on different scales.
pub fn fuse_rrf(
    dense: &[(String, f32)],
    sparse: &[(String, f32)],
    top_k: usize,
    k: f32,
) -> Vec<FusedHit> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f32, Option<f32>, Option<f32>)> = HashMap::new();

    // Dense contributions
    for (rank, (doc_id, score)) in dense.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);
        scores.entry(doc_id.clone())
            .and_modify(|(s, d, _)| { *s += rrf_score; *d = Some(*score); })
            .or_insert((rrf_score, Some(*score), None));
    }

    // Sparse contributions
    for (rank, (doc_id, score)) in sparse.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);
        scores.entry(doc_id.clone())
            .and_modify(|(s, _, sp)| { *s += rrf_score; *sp = Some(*score); })
            .or_insert((rrf_score, None, Some(*score)));
    }

    // Sort by fused score descending
    let mut sorted: Vec<(String, f32, Option<f32>, Option<f32>)> = scores
        .into_iter()
        .map(|(id, (s, d, sp))| (id, s, d, sp))
        .collect();

    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(top_k);

    sorted.into_iter()
        .map(|(doc_id, score, dense_score, sparse_score)| FusedHit {
            doc_id, score, dense_score, sparse_score,
        })
        .collect()
}

/// Fuse using weighted linear combination.
///
/// Dense scores should be normalized to [0, 1] before fusion.
pub fn fuse_weighted(
    dense: &[(String, f32)],
    sparse: &[(String, f32)],
    top_k: usize,
    alpha: f32,
) -> Vec<FusedHit> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f32, Option<f32>, Option<f32>)> = HashMap::new();

    let max_sparse = sparse.first().map(|(_, s)| *s).unwrap_or(1.0).max(1e-6);

    for (doc_id, score) in dense.iter() {
        let w = alpha * score;
        scores.entry(doc_id.clone())
            .and_modify(|(s, d, _)| { *s += w; *d = Some(*score); })
            .or_insert((w, Some(*score), None));
    }

    for (doc_id, score) in sparse.iter() {
        let normalized = score / max_sparse; // Normalize to [0, 1]
        let w = (1.0 - alpha) * normalized;
        scores.entry(doc_id.clone())
            .and_modify(|(s, _, sp)| { *s += w; *sp = Some(*score); })
            .or_insert((w, None, Some(*score)));
    }

    let mut sorted: Vec<_> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(top_k);

    sorted.into_iter()
        .map(|(doc_id, (score, dense_score, sparse_score))| FusedHit {
            doc_id, score, dense_score, sparse_score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let dense = vec![
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.5),
        ];
        let sparse = vec![
            ("c".to_string(), 10.0),
            ("b".to_string(), 8.0),
        ];

        let fused = fuse_rrf(&dense, &sparse, 5, 60.0);
        // "b" appears in both → should rank highest
        assert!(!fused.is_empty());
        assert_eq!(fused[0].doc_id, "b");
    }

    #[test]
    fn test_weighted_fusion() {
        let dense = vec![("a".to_string(), 0.9), ("b".to_string(), 0.1)];
        let sparse = vec![("b".to_string(), 10.0), ("a".to_string(), 1.0)];
        // alpha=0.5: b gets 0.05 + 0.5 = 0.55, a gets 0.45 + 0.05 = 0.50
        let fused = fuse_weighted(&dense, &sparse, 5, 0.5);
        assert!(!fused.is_empty());
    }
}
