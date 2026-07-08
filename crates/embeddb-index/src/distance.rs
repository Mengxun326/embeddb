//! Distance metric implementations with SIMD acceleration.
//!
//! Provides scalar, AVX2 (x86_64), and NEON (aarch64) implementations
//! of cosine, euclidean, and dot product distances. The fastest
//! available implementation is selected at runtime via CPU feature detection.
//!
//! # Implementation Notes
//!
//! - AVX2: 8 f32 lanes per operation, FMA (fused multiply-add) for dot products
//! - NEON: 4 f32 lanes per operation, FMA via vfmaq_f32
//! - Scalar: portable fallback for all platforms

// ---------------------------------------------------------------------------
// Scalar (portable) implementations
// ---------------------------------------------------------------------------

/// Compute cosine distance — scalar fallback.
#[inline]
pub fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
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
        return 1.0;
    }

    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Compute Euclidean distance — scalar fallback.
#[inline]
pub fn euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Compute dot product — scalar fallback.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute squared Euclidean distance — scalar fallback.
#[inline]
pub fn euclidean_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    /// Compute dot product using AVX2 + FMA. Returns (dot, norm_a_sq, norm_b_sq).
    #[inline]
    unsafe fn dot_and_norms_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let n = a.len();
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        let mut i = 0;
        // Process 8 elements at a time
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // dot += a * b
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            // norm_a += a * a
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            // norm_b += b * b
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);

            i += 8;
        }

        // Horizontal sum: extract 4x u64 lanes and reduce
        let dot_scalar = horizontal_sum_avx(dot_sum);
        let norm_a_scalar = horizontal_sum_avx(norm_a_sum);
        let norm_b_scalar = horizontal_sum_avx(norm_b_sum);

        // Remainder
        let mut dot_rem = 0.0f32;
        let mut norm_a_rem = 0.0f32;
        let mut norm_b_rem = 0.0f32;
        while i < n {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
            i += 1;
        }

        (
            dot_scalar + dot_rem,
            norm_a_scalar + norm_a_rem,
            norm_b_scalar + norm_b_rem,
        )
    }

    /// Horizontal sum of 8 f32 values in an AVX register.
    #[inline]
    unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
        // v = [a,b,c,d, e,f,g,h]
        let hi = _mm256_extractf128_ps(v, 1);     // [e,f,g,h]
        let lo = _mm256_castps256_ps128(v);        // [a,b,c,d]
        let sum128 = _mm_add_ps(lo, hi);           // [a+e, b+f, c+g, d+h]
        // hadd: [a+e+b+f, c+g+d+h, a+e+b+f, c+g+d+h]
        let hadd1 = _mm_hadd_ps(sum128, sum128);
        let hadd2 = _mm_hadd_ps(hadd1, hadd1);     // [sum, sum, sum, sum]
        _mm_cvtss_f32(hadd2)
    }

    /// Cosine distance — AVX2-accelerated.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let (dot, norm_a, norm_b) = dot_and_norms_avx2(a, b);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Euclidean distance — AVX2-accelerated.
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = _mm256_setzero_ps();

        let mut i = 0;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        let mut total = horizontal_sum_avx(sum);

        while i < n {
            let diff = a[i] - b[i];
            total += diff * diff;
            i += 1;
        }

        total.sqrt()
    }

    /// Dot product — AVX2-accelerated.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = _mm256_setzero_ps();

        let mut i = 0;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
            i += 8;
        }

        let mut total = horizontal_sum_avx(sum);

        while i < n {
            total += a[i] * b[i];
            i += 1;
        }

        total
    }

    /// Squared Euclidean distance — AVX2-accelerated (no sqrt).
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = _mm256_setzero_ps();

        let mut i = 0;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        let mut total = horizontal_sum_avx(sum);

        while i < n {
            let diff = a[i] - b[i];
            total += diff * diff;
            i += 1;
        }

        total
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    /// Compute dot product using NEON + FMA. Returns (dot, norm_a_sq, norm_b_sq).
    #[inline]
    unsafe fn dot_and_norms_neon(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let n = a.len();
        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        let mut i = 0;
        // Process 4 elements at a time
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            dot_sum = vfmaq_f32(dot_sum, va, vb);
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);

            i += 4;
        }

        let dot_scalar = vaddvq_f32(dot_sum);
        let norm_a_scalar = vaddvq_f32(norm_a_sum);
        let norm_b_scalar = vaddvq_f32(norm_b_sum);

        // Remainder
        let mut dot_rem = 0.0f32;
        let mut norm_a_rem = 0.0f32;
        let mut norm_b_rem = 0.0f32;
        while i < n {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
            i += 1;
        }

        (
            dot_scalar + dot_rem,
            norm_a_scalar + norm_a_rem,
            norm_b_scalar + norm_b_rem,
        )
    }

    /// Cosine distance — NEON-accelerated.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let (dot, norm_a, norm_b) = dot_and_norms_neon(a, b);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Euclidean distance — NEON-accelerated.
    #[target_feature(enable = "neon")]
    pub unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
            i += 4;
        }

        let mut total = vaddvq_f32(sum);

        while i < n {
            let diff = a[i] - b[i];
            total += diff * diff;
            i += 1;
        }

        total.sqrt()
    }

    /// Dot product — NEON-accelerated.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            sum = vfmaq_f32(sum, va, vb);
            i += 4;
        }

        let mut total = vaddvq_f32(sum);

        while i < n {
            total += a[i] * b[i];
            i += 1;
        }

        total
    }

    /// Squared Euclidean distance — NEON-accelerated (no sqrt).
    #[target_feature(enable = "neon")]
    pub unsafe fn euclidean_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
            i += 4;
        }

        let mut total = vaddvq_f32(sum);

        while i < n {
            let diff = a[i] - b[i];
            total += diff * diff;
            i += 1;
        }

        total
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch — selects fastest available implementation
// ---------------------------------------------------------------------------

/// Returns true if AVX2 is supported on the current CPU.
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns true if NEON is supported on the current CPU.
#[inline]
fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// Public dispatch functions
// ---------------------------------------------------------------------------

/// Compute cosine distance between two vectors.
///
/// Returns 1 - cosine_similarity, range [0, 2].
/// Lower values indicate more similar vectors.
///
/// Automatically selects the fastest SIMD implementation available.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return x86_64::cosine_avx2(a, b);
        }
    }
    if has_neon() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return aarch64::cosine_neon(a, b);
        }
    }
    cosine_scalar(a, b)
}

/// Compute Euclidean (L2) distance between two vectors.
///
/// Range [0, ∞). Lower values indicate more similar vectors.
/// Automatically selects the fastest SIMD implementation available.
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return x86_64::euclidean_avx2(a, b);
        }
    }
    if has_neon() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return aarch64::euclidean_neon(a, b);
        }
    }
    euclidean_scalar(a, b)
}

/// Compute dot product between two vectors.
///
/// Range (-∞, ∞). Higher values indicate more similar vectors.
/// (The index layer negates this for consistent ascending sort.)
/// Automatically selects the fastest SIMD implementation available.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return x86_64::dot_product_avx2(a, b);
        }
    }
    if has_neon() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return aarch64::dot_product_neon(a, b);
        }
    }
    dot_product_scalar(a, b)
}

/// Compute squared Euclidean distance (avoids sqrt for comparison).
///
/// Automatically selects the fastest SIMD implementation available.
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return x86_64::euclidean_squared_avx2(a, b);
        }
    }
    if has_neon() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return aarch64::euclidean_squared_neon(a, b);
        }
    }
    euclidean_squared_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test both SIMD and scalar paths on all platforms

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 0.0).abs() < 1e-6);
        assert!((cosine_scalar(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_scalar(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine(&a, &b) - 2.0).abs() < 1e-6);
        assert!((cosine_scalar(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((euclidean(&v, &v) - 0.0).abs() < 1e-6);
        assert!((euclidean_scalar(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-6);
        assert!((euclidean_scalar(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
        assert!((dot_product_scalar(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_long_vectors() {
        // Test with vectors longer than SIMD width
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();

        let cos_simd = cosine(&a, &b);
        let cos_scalar = cosine_scalar(&a, &b);
        assert!((cos_simd - cos_scalar).abs() < 1e-5, "SIMD {cos_simd} != scalar {cos_scalar}");

        let euc_simd = euclidean(&a, &b);
        let euc_scalar = euclidean_scalar(&a, &b);
        assert!((euc_simd - euc_scalar).abs() < 1e-4, "SIMD {euc_simd} != scalar {euc_scalar}");

        let dot_simd = dot_product(&a, &b);
        let dot_scalar = dot_product_scalar(&a, &b);
        assert!((dot_simd - dot_scalar).abs() < 1e-4, "SIMD {dot_simd} != scalar {dot_scalar}");
    }

    #[test]
    fn test_unaligned_vectors() {
        // Vectors with lengths not divisible by SIMD width
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..17).map(|i| (i * 3) as f32).collect();

        let cos_simd = cosine(&a, &b);
        let cos_scalar = cosine_scalar(&a, &b);
        assert!((cos_simd - cos_scalar).abs() < 1e-5);

        let euc_simd = euclidean(&a, &b);
        let euc_scalar = euclidean_scalar(&a, &b);
        assert!((euc_simd - euc_scalar).abs() < 1e-4);
    }
}
