//! # ternary-entropy
//!
//! Information-theoretic measures for three-valued distributions over
//! the ternary alphabet {-1, 0, +1}.
//!
//! This crate provides a `no_std`-compatible core library for computing
//! Shannon entropy, joint entropy, mutual information, conditional entropy,
//! Kullback–Leibler divergence, and cross-entropy for ternary probability
//! distributions.
//!
//! ## Quick Start
//!
//! ```rust
//! use ternary_entropy::*;
//!
//! let probs = [0.2, 0.5, 0.3];
//! let h = ternary_shannon_entropy(probs);
//! assert!((h - 1.029_65).abs() < 1e-4);
//!
//! let h_bits = ternary_shannon_entropy_bits(probs);
//! assert!((h_bits - 1.485_47).abs() < 1e-4);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
use std::f64;
#[cfg(not(feature = "std"))]
use core::f64;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ---------- Float math abstraction ----------

#[cfg(feature = "std")]
#[inline(always)]
fn float_ln(x: f64) -> f64 { x.ln() }

#[cfg(not(feature = "std"))]
#[inline(always)]
fn float_ln(x: f64) -> f64 { libm::log(x) }

// ---------- Helpers ----------

/// Compute `p * ln(p)` for Shannon entropy, treating `p = 0` as `0`.
#[inline]
fn p_log_p(p: f64) -> f64 {
    if p <= 0.0 { 0.0 } else { p * float_ln(p) }
}

// ---------- TernaryDist ----------

/// A probability distribution over the ternary alphabet {-1, 0, +1}.
///
/// Index `0` → value `-1`, index `1` → value `0`, index `2` → value `+1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernaryDist {
    /// Probabilities indexed as `[-1, 0, +1]`.
    pub probs: [f64; 3],
}

impl TernaryDist {
    /// Create a new normalized ternary distribution from raw weights.
    ///
    /// The weights are normalised so they sum to `1.0`.  If all weights are
    /// zero (or negative) an [`Err`] is returned.
    #[inline]
    pub fn from_weights(weights: [f64; 3]) -> Result<Self, EntropyError> {
        let sum: f64 = weights.iter().copied().sum();
        if sum <= 0.0 {
            return Err(EntropyError::ZeroTotalWeight);
        }
        let norm = sum.recip();
        Ok(Self { probs: [weights[0] * norm, weights[1] * norm, weights[2] * norm] })
    }

    /// Create a new ternary distribution directly from probabilities.
    ///
    /// Returns [`Err`] if any probability is negative, greater than `1.0`, or
    /// if the sum is not within `1e-12` of `1.0`.
    #[inline]
    pub fn new(probs: [f64; 3]) -> Result<Self, EntropyError> {
        for &p in &probs {
            if p < 0.0 { return Err(EntropyError::NegativeProbability(p)); }
            if p > 1.0 { return Err(EntropyError::ProbabilityExceedsUnity(p)); }
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-12 {
            return Err(EntropyError::InvalidSum(sum));
        }
        Ok(Self { probs })
    }

    /// Returns the probability for ternary symbol `value` (-1, 0, or +1).
    ///
    /// # Panics
    ///
    /// Panics if `value` is not in {-1, 0, +1}.
    pub fn prob(&self, value: i8) -> f64 {
        match value {
            -1 => self.probs[0],
            0 => self.probs[1],
            1 => self.probs[2],
            other => panic!("Invalid ternary value: {}. Must be -1, 0, or +1.", other),
        }
    }

    /// Shannon entropy in nats (base *e*).
    ///
    /// ```math
    /// H(X) = -\sum_{x \in \{-1,0,+1\}} p(x) \ln p(x)
    /// ```
    #[inline]
    pub fn entropy_nats(&self) -> f64 { entropy_nats(self.probs) }

    /// Shannon entropy in bits (base 2).
    #[inline]
    pub fn entropy_bits(&self) -> f64 { entropy_bits(self.probs) }

    /// Shannon entropy in trits (base 3).
    #[inline]
    pub fn entropy_trits(&self) -> f64 { entropy_trits(self.probs) }
}

// ---------- Error type ----------

/// Errors returned by [`TernaryDist::new`] and [`TernaryDist::from_weights`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntropyError {
    /// A probability was negative.
    NegativeProbability(f64),
    /// A probability exceeded 1.0.
    ProbabilityExceedsUnity(f64),
    /// The probabilities do not sum to 1.0.
    InvalidSum(f64),
    /// All weights were zero or negative (cannot normalise).
    ZeroTotalWeight,
    /// A dimension mismatch occurred during joint operations.
    DimensionMismatch,
}

#[cfg(feature = "std")]
impl std::fmt::Display for EntropyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeProbability(p) => write!(f, "negative probability: {}", p),
            Self::ProbabilityExceedsUnity(p) => write!(f, "probability > 1.0: {}", p),
            Self::InvalidSum(s) => write!(f, "probabilities sum to {}, expected 1.0", s),
            Self::ZeroTotalWeight => write!(f, "all weights are zero or negative"),
            Self::DimensionMismatch => write!(f, "dimension mismatch in joint operation"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EntropyError {}

// ---------- Core entropy functions ----------

/// Shannon entropy in nats (base *e*) for a 3-outcome distribution.
///
/// ```math
/// H(X) = -\sum_{i=1}^{3} p_i \ln p_i
/// ```
///
/// By convention, `0 \cdot \ln(0) \equiv 0`.
#[inline]
pub fn ternary_shannon_entropy(probs: [f64; 3]) -> f64 { entropy_nats(probs) }

/// Shannon entropy in bits (base 2) for a 3-outcome distribution.
///
/// ```math
/// H_{\text{bits}}(X) = -\sum_{i=1}^{3} p_i \log_2 p_i
/// ```
#[inline]
pub fn ternary_shannon_entropy_bits(probs: [f64; 3]) -> f64 { entropy_bits(probs) }

/// Shannon entropy in trits (base 3) for a 3-outcome distribution.
///
/// The maximum entropy for a ternary distribution is 1 trit (when uniform).
#[inline]
pub fn ternary_shannon_entropy_trits(probs: [f64; 3]) -> f64 { entropy_trits(probs) }

fn entropy_nats(probs: [f64; 3]) -> f64 {
    -(p_log_p(probs[0]) + p_log_p(probs[1]) + p_log_p(probs[2]))
}

fn entropy_bits(probs: [f64; 3]) -> f64 {
    -(p_log_p(probs[0]) + p_log_p(probs[1]) + p_log_p(probs[2])) / float_ln(2.0)
}

fn entropy_trits(probs: [f64; 3]) -> f64 {
    -(p_log_p(probs[0]) + p_log_p(probs[1]) + p_log_p(probs[2])) / float_ln(3.0)
}

// ---------- Joint entropy ----------

/// Joint entropy `H(X, Y)` for a joint distribution over two ternary alphabets.
///
/// The joint distribution is a 3×3 matrix `joint[i][j]` where `i` indexes
/// the outcome of `X` and `j` indexes the outcome of `Y`.
///
/// ```math
/// H(X, Y) = -\sum_{x}\sum_{y} p(x, y) \ln p(x, y)
/// ```
pub fn joint_entropy(joint: [[f64; 3]; 3]) -> f64 {
    let mut h = 0.0;
    for row in &joint {
        for &p in row {
            h -= p_log_p(p);
        }
    }
    h
}

// ---------- Mutual information ----------

/// Mutual information `I(X; Y)` for two ternary distributions and their joint.
///
/// Returns `I(X; Y) = H(X) + H(Y) - H(X, Y)`.
///
/// ```math
/// I(X; Y) = \sum_{x}\sum_{y} p(x, y) \ln\frac{p(x, y)}{p(x)\,p(y)}
/// ```
pub fn mutual_information(px: [f64; 3], py: [f64; 3], joint: [[f64; 3]; 3]) -> f64 {
    let hx = entropy_nats(px);
    let hy = entropy_nats(py);
    let hxy = joint_entropy(joint);
    let mi = hx + hy - hxy;
    if mi < 0.0 && mi > -1e-12 { 0.0 } else { mi }
}

// ---------- Conditional entropy ----------

/// Conditional entropy `H(X | Y)` in nats.
///
/// ```math
/// H(X | Y) = H(X, Y) - H(Y)
/// ```
pub fn conditional_entropy(_px: [f64; 3], py: [f64; 3], joint: [[f64; 3]; 3]) -> f64 {
    joint_entropy(joint) - entropy_nats(py)
}

/// Conditional entropy `H(Y | X)` in nats.
///
/// ```math
/// H(Y | X) = H(X, Y) - H(X)
/// ```
pub fn conditional_entropy_y_given_x(
    px: [f64; 3],
    _py: [f64; 3],
    joint: [[f64; 3]; 3],
) -> f64 {
    joint_entropy(joint) - entropy_nats(px)
}

// ---------- KL divergence ----------

/// Kullback–Leibler divergence `D_KL(P ‖ Q)` for two ternary distributions.
///
/// ```math
/// D_{\text{KL}}(P \| Q) = \sum_{i=1}^{3} p_i \ln\frac{p_i}{q_i}
/// ```
///
/// Uses the convention `0 \cdot \ln(0 / q) \equiv 0` and
/// `p \cdot \ln(p / 0) = \infty` (returns `f64::INFINITY`).
pub fn kl_divergence(p: [f64; 3], q: [f64; 3]) -> f64 {
    let mut d = 0.0;
    for i in 0..3 {
        let pi = p[i];
        let qi = q[i];
        if pi <= 0.0 { continue; }
        if qi <= 0.0 { return f64::INFINITY; }
        d += pi * float_ln(pi / qi);
    }
    d
}

// ---------- Cross entropy ----------

/// Cross-entropy `H(p, q)` in nats for two ternary distributions.
///
/// ```math
/// H(p, q) = -\sum_{i=1}^{3} p_i \ln q_i
/// ```
///
/// This is the expected negative log-likelihood when using `q` to approximate
/// `p`.  Zero `q_i` with non-zero `p_i` produce `f64::INFINITY`.
pub fn ternary_cross_entropy(p: [f64; 3], q: [f64; 3]) -> f64 {
    let mut ce = 0.0;
    for i in 0..3 {
        let pi = p[i];
        let qi = q[i];
        if pi <= 0.0 { continue; }
        if qi <= 0.0 { return f64::INFINITY; }
        ce -= pi * float_ln(qi);
    }
    ce
}

// ---------- Sliding-window entropy ----------

/// Compute local Shannon entropy (nats) over a sliding window of a ternary sequence.
///
/// Each window yields the entropy `H(W)` of the empirical distribution within
/// that window. The result vector has length `sequence.len() - window_size + 1`
/// (0 if the sequence is shorter than the window).
pub fn sliding_window_entropy(sequence: &[i8], window_size: usize) -> Vec<f64> {
    if sequence.len() < window_size || window_size == 0 {
        return Vec::new();
    }
    let n = sequence.len() - window_size + 1;
    let mut result = Vec::with_capacity(n);
    let mut counts = [0usize; 3];

    for &s in &sequence[..window_size] {
        counts[symbol_to_index(s)] += 1;
    }

    let win_f = window_size as f64;
    result.push(window_entropy(&counts, win_f));

    for i in window_size..sequence.len() {
        counts[symbol_to_index(sequence[i - window_size])] -= 1;
        counts[symbol_to_index(sequence[i])] += 1;
        result.push(window_entropy(&counts, win_f));
    }

    result
}

#[inline(always)]
fn symbol_to_index(s: i8) -> usize {
    match s { -1 => 0, 0 => 1, 1 => 2, other => panic!("invalid ternary symbol: {}", other) }
}

#[inline]
fn window_entropy(counts: &[usize; 3], total: f64) -> f64 {
    let mut h = 0.0;
    for &c in counts {
        if c == 0 { continue; }
        let p = c as f64 / total;
        h -= p * float_ln(p);
    }
    h
}

/// Shannon entropy `H(X)` in nats for a [`TernaryDist`].
#[inline]
pub fn entropy(dist: &TernaryDist) -> f64 { entropy_nats(dist.probs) }

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use core::f64::consts::LN_2;

    // Import the float_ln for tests
    use super::*;

    // --- TernaryDist construction ---

    #[test]
    fn test_new_valid() {
        let d = TernaryDist::new([0.2, 0.5, 0.3]).unwrap();
        assert!((d.probs[0] - 0.2).abs() < 1e-12);
        assert!((d.probs[1] - 0.5).abs() < 1e-12);
        assert!((d.probs[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_new_negative_probability() {
        assert_eq!(
            TernaryDist::new([-0.1, 0.6, 0.5]),
            Err(EntropyError::NegativeProbability(-0.1))
        );
    }

    #[test]
    fn test_new_probability_exceeds_unity() {
        assert_eq!(
            TernaryDist::new([1.5, 0.0, -0.5]),
            Err(EntropyError::ProbabilityExceedsUnity(1.5))
        );
    }

    #[test]
    fn test_new_invalid_sum() {
        let err = TernaryDist::new([0.2, 0.3, 0.4]).unwrap_err();
        assert!(matches!(err, EntropyError::InvalidSum(s) if (s - 0.9).abs() < 1e-12));
    }

    #[test]
    fn test_from_weights_zero_total() {
        assert_eq!(
            TernaryDist::from_weights([0.0, 0.0, 0.0]),
            Err(EntropyError::ZeroTotalWeight)
        );
    }

    #[test]
    fn test_fractional_weights_normalised() {
        let d = TernaryDist::from_weights([1.0, 2.0, 3.0]).unwrap();
        assert!((d.probs[0] - 1.0 / 6.0).abs() < 1e-12);
        assert!((d.probs[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((d.probs[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_prob_accessor() {
        let d = TernaryDist::new([0.1, 0.7, 0.2]).unwrap();
        assert!((d.prob(-1) - 0.1).abs() < 1e-12);
        assert!((d.prob(0) - 0.7).abs() < 1e-12);
        assert!((d.prob(1) - 0.2).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_prob_invalid() {
        let d = TernaryDist::new([0.2, 0.5, 0.3]).unwrap();
        let _ = d.prob(42);
    }

    #[test]
    fn test_entropy_method_nats() {
        let d = TernaryDist::new([0.5, 0.5, 0.0]).unwrap();
        assert!((d.entropy_nats() - 0.693147).abs() < 1e-5);
    }

    #[test]
    fn test_entropy_method_bits() {
        let d = TernaryDist::new([0.5, 0.5, 0.0]).unwrap();
        assert!((d.entropy_bits() - 1.0).abs() < 1e-10);
    }

    // --- Zero probability edge cases ---

    #[test]
    fn test_deterministic_zero_entropy() {
        assert!((ternary_shannon_entropy([1.0, 0.0, 0.0])).abs() < 1e-12);
    }

    #[test]
    fn test_deterministic_bits() {
        assert!((ternary_shannon_entropy_bits([0.0, 1.0, 0.0])).abs() < 1e-12);
    }

    #[test]
    fn test_two_outcomes_only() {
        let h = ternary_shannon_entropy([0.5, 0.0, 0.5]);
        assert!((h - LN_2).abs() < 1e-12);
    }

    #[test]
    fn test_all_mass_on_middle() {
        assert!((ternary_shannon_entropy([0.0, 1.0, 0.0])).abs() < 1e-12);
    }

    // --- Uniform distribution ---

    #[test]
    fn test_uniform_nats() {
        let u = 1.0 / 3.0;
        assert!((ternary_shannon_entropy([u, u, u]) - 3.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_bits() {
        let u = 1.0 / 3.0;
        let h = ternary_shannon_entropy_bits([u, u, u]);
        assert!((h - (3.0_f64.ln() / LN_2)).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_trits() {
        let u = 1.0 / 3.0;
        assert!((ternary_shannon_entropy_trits([u, u, u]) - 1.0).abs() < 1e-12);
    }

    // --- Joint entropy ---

    #[test]
    fn test_joint_independent() {
        let joint = [[1.0 / 9.0; 3]; 3];
        let hxy = joint_entropy(joint);
        let hx = ternary_shannon_entropy([1.0 / 3.0; 3]);
        assert!((hxy - 2.0 * hx).abs() < 1e-12);
    }

    #[test]
    fn test_joint_deterministic() {
        let joint = [
            [0.4, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let hxy = joint_entropy(joint);
        let hx = ternary_shannon_entropy([0.4, 0.35, 0.25]);
        assert!((hxy - hx).abs() < 1e-12);
    }

    #[test]
    fn test_joint_with_zeros() {
        let joint = [
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ];
        let hxy = joint_entropy(joint);
        let hx = ternary_shannon_entropy([0.6, 0.0, 0.4]);
        assert!((hxy - hx).abs() < 1e-12);
    }

    // --- Mutual information ---

    #[test]
    fn test_mi_independent_zero() {
        let px = [1.0 / 3.0; 3];
        let py = [1.0 / 3.0; 3];
        let joint = [[1.0 / 9.0; 3]; 3];
        let mi = mutual_information(px, py, joint);
        assert!(mi.abs() < 1e-10);
    }

    #[test]
    fn test_mi_perfect_correlation() {
        let px = [0.4, 0.35, 0.25];
        let py = [0.4, 0.35, 0.25];
        let joint = [
            [0.4, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let mi = mutual_information(px, py, joint);
        let hx = ternary_shannon_entropy(px);
        assert!((mi - hx).abs() < 1e-12);
    }

    #[test]
    fn test_mi_symmetric() {
        let px = [0.5, 0.3, 0.2];
        let py = [0.4, 0.4, 0.2];
        let joint = [
            [0.25, 0.15, 0.10],
            [0.10, 0.20, 0.10],
            [0.05, 0.05, 0.00],
        ];
        let mi_xy = mutual_information(px, py, joint);
        let mi_yx = mutual_information(py, px, transpose(joint));
        assert!((mi_xy - mi_yx).abs() < 1e-12);
    }

    fn transpose(j: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        [
            [j[0][0], j[1][0], j[2][0]],
            [j[0][1], j[1][1], j[2][1]],
            [j[0][2], j[1][2], j[2][2]],
        ]
    }

    // --- Conditional entropy ---

    #[test]
    fn test_ce_independent() {
        let px = [1.0 / 3.0; 3];
        let py = [1.0 / 3.0; 3];
        let joint = [[1.0 / 9.0; 3]; 3];
        let ce = conditional_entropy(px, py, joint);
        let hx = ternary_shannon_entropy(px);
        assert!((ce - hx).abs() < 1e-12);
    }

    #[test]
    fn test_ce_deterministic() {
        let py = [0.4, 0.35, 0.25];
        let joint = [
            [0.4, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let ce = conditional_entropy(py, py, joint);
        assert!(ce.abs() < 1e-12);
    }

    #[test]
    fn test_ce_chain_rule() {
        let px = [0.5, 0.3, 0.2];
        let py = [0.4, 0.4, 0.2];
        let joint = [
            [0.25, 0.15, 0.10],
            [0.10, 0.20, 0.10],
            [0.05, 0.05, 0.00],
        ];
        let hxy = joint_entropy(joint);
        let hx_given_y = conditional_entropy(px, py, joint);
        let hy_given_x = conditional_entropy_y_given_x(px, py, joint);
        assert!((hxy - (entropy_nats(py) + hx_given_y)).abs() < 1e-12);
        assert!((hxy - (entropy_nats(px) + hy_given_x)).abs() < 1e-12);
    }

    // --- KL divergence ---

    #[test]
    fn test_kl_identical() {
        let p = [0.3, 0.4, 0.3];
        assert!((kl_divergence(p, p)).abs() < 1e-12);
    }

    #[test]
    fn test_kl_nonnegative() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.5, 0.4, 0.1];
        assert!(kl_divergence(p, q) >= 0.0);
    }

    #[test]
    fn test_kl_infinite_when_q_zero() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.5, 0.5, 0.0];
        assert_eq!(kl_divergence(p, q), f64::INFINITY);
    }

    #[test]
    fn test_kl_zero_p_zero() {
        let p = [0.5, 0.5, 0.0];
        let q = [0.5, 0.5, 0.0];
        assert!((kl_divergence(p, q)).abs() < 1e-12);
    }

    // --- Cross entropy ---

    #[test]
    fn test_ce_identity() {
        let p = [0.2, 0.5, 0.3];
        let ce = ternary_cross_entropy(p, p);
        assert!((ce - ternary_shannon_entropy(p)).abs() < 1e-12);
    }

    #[test]
    fn test_ce_plus_kl() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.5, 0.4, 0.1];
        let ce = ternary_cross_entropy(p, q);
        let hp = ternary_shannon_entropy(p);
        let kl = kl_divergence(p, q);
        assert!((ce - (hp + kl)).abs() < 1e-12);
    }

    #[test]
    fn test_ce_infinite() {
        let p = [0.5, 0.3, 0.2];
        let q = [0.5, 0.5, 0.0];
        assert_eq!(ternary_cross_entropy(p, q), f64::INFINITY);
    }

    // --- Sliding window ---

    #[test]
    fn test_sliding_basic() {
        let seq = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, 1];
        let entropies = sliding_window_entropy(&seq, 3);
        assert_eq!(entropies.len(), 8);
        for &h in &entropies {
            assert!(h >= 0.0);
            assert!(h <= 3.0_f64.ln() + 1e-12);
        }
    }

    #[test]
    fn test_sliding_all_same() {
        let seq = vec![1, 1, 1, 1, 1];
        let entropies = sliding_window_entropy(&seq, 3);
        assert_eq!(entropies.len(), 3);
        for &h in &entropies {
            assert!((h).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sliding_empty() {
        let seq: Vec<i8> = vec![];
        let entropies = sliding_window_entropy(&seq, 3);
        assert_eq!(entropies.len(), 0);
    }

    #[test]
    fn test_sliding_window_larger_than_seq() {
        let seq = vec![1, 0, -1];
        let entropies = sliding_window_entropy(&seq, 5);
        assert_eq!(entropies.len(), 0);
    }

    #[test]
    fn test_sliding_window_size_zero() {
        let seq = vec![1, 0, -1];
        let entropies = sliding_window_entropy(&seq, 0);
        assert_eq!(entropies.len(), 0);
    }

    #[test]
    fn test_sliding_window_exact() {
        let seq = vec![1, 1, 1, 0, 0, 0, -1, -1, -1];
        let entropies = sliding_window_entropy(&seq, 9);
        assert_eq!(entropies.len(), 1);
        let h = entropies[0];
        let expected = ternary_shannon_entropy([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert!((h - expected).abs() < 1e-12);
    }

    // --- Free function entropy ---

    #[test]
    fn test_free_entropy_fn() {
        let d = TernaryDist::new([0.5, 0.5, 0.0]).unwrap();
        assert!((entropy(&d) - 0.693147).abs() < 1e-5);
    }

    // --- Known values regression ---

    #[test]
    fn test_known_values() {
        let h = ternary_shannon_entropy([0.2, 0.5, 0.3]);
        let expected = -(0.2 * float_ln(0.2) + 0.5 * float_ln(0.5) + 0.3 * float_ln(0.3));
        assert!((h - expected).abs() < 1e-12);

        let h_bits = ternary_shannon_entropy_bits([0.2, 0.5, 0.3]);
        assert!((h_bits - expected / LN_2).abs() < 1e-12);
    }
}
