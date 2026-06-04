# ternary-entropy: Information-theoretic measures for three-valued distributions

Shannon entropy, conditional entropy, mutual information, KL divergence, and sliding-window analysis over probability distributions whose outcomes are positive (+1), negative (−1), or neutral (0).

## Why This Exists

When you model decisions as three-valued signals—buy/hold/sell, agree/neutral/disagree, accelerate/coast/brake—you need to answer questions like "how uncertain is this signal?" and "how much does knowing one signal tell me about another?" Standard binary entropy tools don't fit. This crate provides the full information-theory toolkit purpose-built for ternary distributions, including joint distributions, conditional entropy, and sliding-window entropy traces.

## Core Concepts

- **Ternary** — A value that is `Positive` (+1), `Negative` (−1), or `Neutral` (0). Not a boolean; not a float. Three states.
- **TernaryDistribution** — A probability distribution over the three ternary values. Stores `[P(Positive), P(Negative), P(Neutral)]`. Sum must equal 1.0.
- **Shannon entropy** — A measure of unpredictability. For a ternary distribution, it ranges from 0 (completely deterministic) to log₂(3) ≈ 1.585 bits (uniform).
- **JointDistribution** — A 3×3 probability table for pairs of ternary variables. Used to compute how two signals relate.
- **Conditional entropy H(B|A)** — How uncertain B remains after you know A. If H(B|A) = 0, then A completely determines B.
- **Mutual information I(A;B)** — How many bits of information A and B share. Zero means they're independent.
- **KL divergence D_KL(P‖Q)** — How different distribution P is from Q, measured in bits. Not symmetric; P is the "true" distribution, Q is the "model."
- **Jensen-Shannon divergence** — A symmetric, smoothed version of KL divergence. Always finite. Bounded by log₂(3) for ternary distributions.
- **Entropy rate** — The per-symbol uncertainty of a sequence, computed from conditional entropy of consecutive pairs.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-entropy = "0.1"
```

```rust
use ternary_entropy::*;

// Build a distribution from observed data
let seq = vec![
    Ternary::Positive, Ternary::Positive, Ternary::Negative,
    Ternary::Neutral, Ternary::Positive,
];
let dist = TernaryDistribution::from_sequence(&seq);

// How uncertain is this signal?
let h = shannon_entropy(&dist);
println!("Entropy: {:.3} bits (max: {:.3})", h, max_entropy());

// Normalized to 0..1 for comparison across contexts
let norm = normalized_entropy(&dist);
println!("Normalized: {:.3}", norm);

// Compare two distributions
let uniform = TernaryDistribution::uniform();
let kl = kl_divergence(&dist, &uniform);
println!("KL divergence from uniform: {:.3} bits", kl);

// Sliding-window entropy trace
let trace = sliding_entropy(&seq, 3);
println!("Entropy over windows of 3: {:?}", trace);
```

## API Overview

| Type / Function | What it is |
|---|---|
| `Ternary` | Enum: `Positive`, `Negative`, `Neutral` |
| `TernaryDistribution` | Probability distribution over the three values |
| `JointDistribution` | 3×3 joint probability table for pairs |
| `shannon_entropy` | H(X) in bits |
| `max_entropy` | log₂(3) ≈ 1.585 bits |
| `normalized_entropy` | H(X) / log₂(3), in range [0, 1] |
| `conditional_entropy` | H(B\|A) from a joint distribution |
| `joint_entropy` | H(A, B) from a joint distribution |
| `mutual_information` | I(A; B) = H(A) + H(B) − H(A, B) |
| `kl_divergence` | D_KL(P\|Q), asymmetric, in bits |
| `js_divergence` | Symmetric divergence, always finite |
| `cross_entropy` | H(P, Q) = −Σ P(x) log₂ Q(x) |
| `sliding_entropy` | Entropy computed over rolling windows |
| `entropy_rate` | Per-symbol uncertainty from sequential data |

## How It Works

**Distribution construction.** `TernaryDistribution::from_sequence` counts occurrences of each ternary value and normalizes by the total. An empty sequence defaults to uniform `[⅓, ⅓, ⅓]`. `TernaryDistribution::new` validates that probabilities sum to 1.0 within a tolerance of 1e-10.

**Entropy calculation.** Shannon entropy uses the standard formula H = −Σ p(x) log₂ p(x), skipping zero-probability terms to avoid −0·log(0).

**Joint distributions.** `JointDistribution::from_pairs` builds a 3×3 contingency table from observed (A, B) pairs. Marginals are computed by summing rows or columns. Conditional probabilities P(B|A) divide the joint cell by the marginal of A.

**Divergences.** KL divergence skips terms where either P or Q is zero (convention: 0·log(0) = 0). Jensen-Shannon uses the midpoint distribution M = (P+Q)/2 and averages KL(M‖P) and KL(M‖Q).

**Sliding entropy.** Applies a fixed-size window across the sequence and computes entropy at each position. Returns one value per window position.

**Entropy rate.** Computes the conditional entropy H(X_t | X_{t−1}) from consecutive pairs, giving the per-step unpredictability of the sequence.

## Known Limitations

- **No smoothing or Laplace correction.** When building distributions from sparse data (few observations), zero-probability entries can make KL divergence undefined (returns a partial sum, skipping those terms). If you need smoothed estimates, pre-process your counts before constructing distributions.
- **Floating-point precision.** Equality checks on probabilities use a tolerance of 1e-10. Very small probabilities (< 1e-15) may suffer from underflow in log computations.
- **Sliding window is O(n·w).** Each window recomputes the distribution from scratch. For large sequences with large windows, this could be optimized with incremental updates, but it isn't yet.

## Use Cases

- **Trading signal analysis.** Compute the entropy of buy/sell/hold signals over time. Rising entropy indicates increasing market uncertainty. Mutual information between two indicators reveals redundant vs. complementary signals.
- **Survey response profiling.** Analyze agree/neutral/disagree distributions across demographics. KL divergence quantifies how different one group's opinion distribution is from another's.
- **Agent strategy evaluation.** Track the entropy rate of an agent's ternary decisions. A decreasing entropy rate indicates the agent is settling into a predictable pattern.

## Ecosystem Context

This is a standalone analysis crate with no dependencies on other ternary crates. It is typically used downstream by `ternary-metrics` (for reporting) and `ternary-scoring` (for incorporating information-theoretic measures into strategy rankings).

## License

MIT
