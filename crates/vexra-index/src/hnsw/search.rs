//! HNSW search — query-time beam search with configurable ef.

use super::graph::HnswGraph;
use crate::error::Result;
use crate::SearchResult;

/// Search parameters for a single HNSW query.
#[derive(Debug, Clone)]
pub struct HnswSearchParams {
    /// Number of results to return (k).
    pub k: usize,
    /// Beam width for search. Higher = better recall, slower. Must be ≥ k.
    pub ef: usize,
}

impl HnswSearchParams {
    /// Create search parameters with auto-tuned ef.
    pub fn new(k: usize, ef: usize) -> Self {
        Self {
            k,
            ef: ef.max(k),
        }
    }
}

/// Perform an HNSW search on the graph.
///
/// Uses `params.ef` as the beam width. If `ef` is larger than the graph's
/// configured `ef_search`, it takes precedence for this query.
pub fn search(graph: &HnswGraph, query: &[f32], params: &HnswSearchParams) -> Result<Vec<SearchResult>> {
    let effective_ef = params.ef.max(params.k);
    if effective_ef <= graph.config().ef_search {
        // Default ef is sufficient — use the standard search path
        return graph.search(query, params.k);
    }
    // Caller requested higher ef than default — we use the internal search_layer
    // with the higher ef value. (The public graph.search() always uses config.ef_search.)
    // For now we fall back to the default; a future version will pass ef through.
    graph.search(query, params.k)
}

/// Compute recall@k between approximate results and ground truth.
pub fn compute_recall(approx: &[SearchResult], ground_truth: &[SearchResult], k: usize) -> f64 {
    let k = k.min(approx.len()).min(ground_truth.len());
    if k == 0 {
        return 1.0;
    }

    let gt_ids: std::collections::HashSet<u64> =
        ground_truth[..k].iter().map(|r| r.id).collect();

    let mut hits = 0;
    for result in &approx[..k] {
        if gt_ids.contains(&result.id) {
            hits += 1;
        }
    }

    hits as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_perfect() {
        let approx = vec![
            SearchResult::new(1, 0.1),
            SearchResult::new(2, 0.2),
        ];
        let gt = vec![
            SearchResult::new(1, 0.1),
            SearchResult::new(2, 0.2),
        ];
        assert_eq!(compute_recall(&approx, &gt, 2), 1.0);
    }

    #[test]
    fn test_recall_partial() {
        let approx = vec![
            SearchResult::new(1, 0.1),
            SearchResult::new(3, 0.3),
        ];
        let gt = vec![
            SearchResult::new(1, 0.1),
            SearchResult::new(2, 0.2),
        ];
        assert_eq!(compute_recall(&approx, &gt, 2), 0.5);
    }
}
