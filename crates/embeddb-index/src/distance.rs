//! Distance metric implementations with scalar fallback.
//!
//! Phase 1 will add SIMD-accelerated (AVX2/NEON) kernels behind
//! architecture feature gates.

/// Compute cosine distance between two vectors.
///
/// Returns 1 - cosine_similarity, range [0, 2].
/// Lower values indicate more similar vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Undefined for zero vectors; treat as maximally distant
    }

    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Compute Euclidean (L2) distance between two vectors.
///
/// Range [0, ∞). Lower values indicate more similar vectors.
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

/// Compute dot product between two vectors.
///
/// Range (-∞, ∞). Higher values indicate more similar vectors.
/// (The index layer negates this for consistent ascending sort.)
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

/// Compute squared Euclidean distance (avoids sqrt for comparison).
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((euclidean(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6); // 4+10+18=32
    }
}
