//! Entropy analysis for ternary strategy distributions.
//!
//! Provides Shannon entropy, conditional entropy, mutual information,
//! and KL divergence calculations.

#![forbid(unsafe_code)]


/// Ternary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Positive,
    Negative,
    Neutral,
}

impl Ternary {
    pub fn value(&self) -> i8 {
        match self {
            Ternary::Positive => 1,
            Ternary::Negative => -1,
            Ternary::Neutral => 0,
        }
    }

    pub fn all() -> [Ternary; 3] {
        [Ternary::Positive, Ternary::Negative, Ternary::Neutral]
    }
}

/// A probability distribution over ternary values.
#[derive(Debug, Clone)]
pub struct TernaryDistribution {
    probs: [f64; 3], // [Positive, Negative, Neutral]
}

impl TernaryDistribution {
    /// Create from explicit probabilities.
    pub fn new(p_pos: f64, p_neg: f64, p_neu: f64) -> Option<Self> {
        let sum = p_pos + p_neg + p_neu;
        if sum <= 0.0 || (sum - 1.0).abs() > 1e-10 {
            return None;
        }
        if p_pos < 0.0 || p_neg < 0.0 || p_neu < 0.0 {
            return None;
        }
        Some(TernaryDistribution { probs: [p_pos, p_neg, p_neu] })
    }

    /// Create a uniform distribution.
    pub fn uniform() -> Self {
        TernaryDistribution { probs: [1.0 / 3.0; 3] }
    }

    /// Create from a sequence of ternary values.
    pub fn from_sequence(seq: &[Ternary]) -> Self {
        let mut counts = [0usize; 3];
        for t in seq {
            match t {
                Ternary::Positive => counts[0] += 1,
                Ternary::Negative => counts[1] += 1,
                Ternary::Neutral => counts[2] += 1,
            }
        }
        let total = seq.len() as f64;
        if total == 0.0 {
            return TernaryDistribution { probs: [1.0 / 3.0; 3] };
        }
        TernaryDistribution {
            probs: [counts[0] as f64 / total, counts[1] as f64 / total, counts[2] as f64 / total],
        }
    }

    /// Get probability of a specific ternary value.
    pub fn prob(&self, t: Ternary) -> f64 {
        match t {
            Ternary::Positive => self.probs[0],
            Ternary::Negative => self.probs[1],
            Ternary::Neutral => self.probs[2],
        }
    }

    /// Get all probabilities as a slice.
    pub fn probs(&self) -> &[f64; 3] {
        &self.probs
    }

    /// Check if this is a valid distribution.
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.probs.iter().sum();
        (sum - 1.0).abs() < 1e-10 && self.probs.iter().all(|&p| p >= 0.0)
    }
}

/// Calculate Shannon entropy of a ternary distribution.
pub fn shannon_entropy(dist: &TernaryDistribution) -> f64 {
    let mut h = 0.0;
    for &p in &dist.probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

/// Maximum entropy for a ternary distribution (log2(3)).
pub fn max_entropy() -> f64 {
    3.0_f64.log2()
}

/// Calculate normalized entropy (0 to 1).
pub fn normalized_entropy(dist: &TernaryDistribution) -> f64 {
    let h = shannon_entropy(dist);
    let max = max_entropy();
    if max == 0.0 { 0.0 } else { h / max }
}

/// A joint distribution over pairs of ternary values.
#[derive(Debug, Clone)]
pub struct JointDistribution {
    /// 3x3 matrix indexed as [first][second]
    probs: [[f64; 3]; 3],
}

impl JointDistribution {
    /// Create from paired observations.
    pub fn from_pairs(pairs: &[(Ternary, Ternary)]) -> Self {
        let mut counts = [[0usize; 3]; 3];
        for (a, b) in pairs {
            let i = idx(*a);
            let j = idx(*b);
            counts[i][j] += 1;
        }
        let total = pairs.len() as f64;
        let mut probs = [[0.0; 3]; 3];
        if total > 0.0 {
            for i in 0..3 {
                for j in 0..3 {
                    probs[i][j] = counts[i][j] as f64 / total;
                }
            }
        }
        JointDistribution { probs }
    }

    /// Create from explicit probabilities.
    pub fn new(probs: [[f64; 3]; 3]) -> Option<Self> {
        let sum: f64 = probs.iter().flat_map(|r| r.iter()).sum();
        if (sum - 1.0).abs() > 1e-10 {
            return None;
        }
        Some(JointDistribution { probs })
    }

    /// Get joint probability.
    pub fn joint_prob(&self, a: Ternary, b: Ternary) -> f64 {
        self.probs[idx(a)][idx(b)]
    }

    /// Marginal distribution of the first variable.
    pub fn marginal_first(&self) -> TernaryDistribution {
        let mut p = [0.0; 3];
        for i in 0..3 {
            p[i] = self.probs[i].iter().sum();
        }
        TernaryDistribution { probs: p }
    }

    /// Marginal distribution of the second variable.
    pub fn marginal_second(&self) -> TernaryDistribution {
        let mut p = [0.0; 3];
        for j in 0..3 {
            p[j] = self.probs.iter().map(|row| row[j]).sum();
        }
        TernaryDistribution { probs: p }
    }

    /// Conditional probability P(B=b | A=a).
    pub fn conditional_prob(&self, a: Ternary, b: Ternary) -> f64 {
        let i = idx(a);
        let pa = self.probs[i].iter().sum::<f64>();
        if pa == 0.0 { 0.0 } else { self.probs[i][idx(b)] / pa }
    }
}

fn idx(t: Ternary) -> usize {
    match t {
        Ternary::Positive => 0,
        Ternary::Negative => 1,
        Ternary::Neutral => 2,
    }
}

/// Calculate conditional entropy H(B|A).
pub fn conditional_entropy(joint: &JointDistribution) -> f64 {
    let mut h = 0.0;
    for a in Ternary::all() {
        let pa = joint.marginal_first().prob(a);
        if pa <= 0.0 { continue; }
        for b in Ternary::all() {
            let p_ba = joint.conditional_prob(a, b);
            if p_ba > 0.0 {
                h -= pa * p_ba * p_ba.log2();
            }
        }
    }
    h
}

/// Calculate joint entropy H(A, B).
pub fn joint_entropy(joint: &JointDistribution) -> f64 {
    let mut h = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let p = joint.probs[i][j];
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
    }
    h
}

/// Calculate mutual information I(A; B).
pub fn mutual_information(joint: &JointDistribution) -> f64 {
    let ha = shannon_entropy(&joint.marginal_first());
    let hb = shannon_entropy(&joint.marginal_second());
    let hab = joint_entropy(joint);
    ha + hb - hab
}

/// Calculate KL divergence D_KL(P || Q).
pub fn kl_divergence(p: &TernaryDistribution, q: &TernaryDistribution) -> f64 {
    let mut d = 0.0;
    for i in 0..3 {
        let pi = p.probs[i];
        let qi = q.probs[i];
        if pi > 0.0 && qi > 0.0 {
            d += pi * (pi / qi).log2();
        }
    }
    d
}

/// Calculate Jensen-Shannon divergence.
pub fn js_divergence(p: &TernaryDistribution, q: &TernaryDistribution) -> f64 {
    let m = TernaryDistribution {
        probs: [
            (p.probs[0] + q.probs[0]) / 2.0,
            (p.probs[1] + q.probs[1]) / 2.0,
            (p.probs[2] + q.probs[2]) / 2.0,
        ],
    };
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// Calculate cross-entropy H(P, Q).
pub fn cross_entropy(p: &TernaryDistribution, q: &TernaryDistribution) -> f64 {
    let mut h = 0.0;
    for i in 0..3 {
        let pi = p.probs[i];
        let qi = q.probs[i];
        if pi > 0.0 && qi > 0.0 {
            h -= pi * qi.log2();
        }
    }
    h
}

/// Analyze entropy of a sequence over sliding windows.
pub fn sliding_entropy(seq: &[Ternary], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > seq.len() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=seq.len() - window_size {
        let window = &seq[i..i + window_size];
        let dist = TernaryDistribution::from_sequence(window);
        result.push(shannon_entropy(&dist));
    }
    result
}

/// Calculate the entropy rate of a sequence.
pub fn entropy_rate(seq: &[Ternary]) -> f64 {
    if seq.len() < 2 {
        return 0.0;
    }
    let pairs: Vec<(Ternary, Ternary)> = seq.windows(2).map(|w| (w[0], w[1])).collect();
    let joint = JointDistribution::from_pairs(&pairs);
    conditional_entropy(&joint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let d = TernaryDistribution::uniform();
        assert!((d.prob(Ternary::Positive) - 1.0 / 3.0).abs() < 1e-10);
        assert!(d.is_valid());
    }

    #[test]
    fn test_from_sequence() {
        let seq = vec![Ternary::Positive, Ternary::Positive, Ternary::Negative];
        let d = TernaryDistribution::from_sequence(&seq);
        assert!((d.prob(Ternary::Positive) - 2.0 / 3.0).abs() < 1e-10);
        assert!((d.prob(Ternary::Negative) - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.prob(Ternary::Neutral)).abs() < 1e-10);
    }

    #[test]
    fn test_from_sequence_empty() {
        let d = TernaryDistribution::from_sequence(&[]);
        assert!(d.is_valid());
    }

    #[test]
    fn test_distribution_new_valid() {
        let d = TernaryDistribution::new(0.5, 0.3, 0.2).unwrap();
        assert!(d.is_valid());
    }

    #[test]
    fn test_distribution_new_invalid() {
        assert!(TernaryDistribution::new(0.5, 0.3, 0.3).is_none()); // sum != 1
        assert!(TernaryDistribution::new(-0.1, 0.5, 0.6).is_none()); // negative
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let d = TernaryDistribution::uniform();
        let h = shannon_entropy(&d);
        assert!((h - 3.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let d = TernaryDistribution::new(1.0, 0.0, 0.0).unwrap();
        assert!((shannon_entropy(&d)).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_entropy() {
        let d = TernaryDistribution::uniform();
        assert!((normalized_entropy(&d) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_entropy() {
        assert!((max_entropy() - 3.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_joint_from_pairs() {
        let pairs = vec![
            (Ternary::Positive, Ternary::Negative),
            (Ternary::Positive, Ternary::Negative),
            (Ternary::Neutral, Ternary::Neutral),
            (Ternary::Neutral, Ternary::Neutral),
        ];
        let j = JointDistribution::from_pairs(&pairs);
        assert!((j.joint_prob(Ternary::Positive, Ternary::Negative) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_marginals() {
        let pairs = vec![
            (Ternary::Positive, Ternary::Negative),
            (Ternary::Negative, Ternary::Positive),
        ];
        let j = JointDistribution::from_pairs(&pairs);
        let m1 = j.marginal_first();
        assert!((m1.prob(Ternary::Positive) - 0.5).abs() < 1e-10);
        assert!((m1.prob(Ternary::Negative) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conditional_entropy_independent() {
        // Independent: H(B|A) = H(B)
        let probs = [[1.0 / 9.0; 3]; 3];
        let j = JointDistribution::new(probs).unwrap();
        let hb = shannon_entropy(&j.marginal_second());
        let hba = conditional_entropy(&j);
        assert!((hba - hb).abs() < 1e-10);
    }

    #[test]
    fn test_joint_entropy() {
        let probs = [[1.0 / 9.0; 3]; 3];
        let j = JointDistribution::new(probs).unwrap();
        let h = joint_entropy(&j);
        assert!((h - (9.0_f64).log2()).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        let probs = [[1.0 / 9.0; 3]; 3];
        let j = JointDistribution::new(probs).unwrap();
        let mi = mutual_information(&j);
        assert!(mi.abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_dependent() {
        let probs = [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]];
        let j = JointDistribution { probs };
        let mi = mutual_information(&j);
        assert!(mi > 1.0); // Perfect correlation
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = TernaryDistribution::uniform();
        assert!(kl_divergence(&d, &d).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = TernaryDistribution::new(0.9, 0.05, 0.05).unwrap();
        let q = TernaryDistribution::uniform();
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = TernaryDistribution::new(0.9, 0.05, 0.05).unwrap();
        let q = TernaryDistribution::new(0.05, 0.9, 0.05).unwrap();
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert!((js_pq - js_qp).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = TernaryDistribution::uniform();
        let q = TernaryDistribution::uniform();
        let h = cross_entropy(&p, &q);
        assert!((h - shannon_entropy(&p)).abs() < 1e-10);
    }

    #[test]
    fn test_sliding_entropy() {
        let seq = vec![Ternary::Positive; 10];
        let result = sliding_entropy(&seq, 3);
        assert_eq!(result.len(), 8);
        for h in result {
            assert!(h.abs() < 1e-10); // All same value = 0 entropy
        }
    }

    #[test]
    fn test_sliding_entropy_empty() {
        let seq: Vec<Ternary> = vec![];
        let result = sliding_entropy(&seq, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_entropy_rate() {
        let seq = vec![Ternary::Positive, Ternary::Positive, Ternary::Positive];
        let rate = entropy_rate(&seq);
        assert!(rate.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_rate_short() {
        let seq = vec![Ternary::Positive];
        assert_eq!(entropy_rate(&seq), 0.0);
    }

    #[test]
    fn test_conditional_prob() {
        let probs = [[0.2, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.2]];
        let j = JointDistribution::new(probs).unwrap();
        let cp = j.conditional_prob(Ternary::Positive, Ternary::Negative);
        let expected = 0.1 / 0.3;
        assert!((cp - expected).abs() < 1e-10);
    }
}
