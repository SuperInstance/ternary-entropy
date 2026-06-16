# CROSS-POLLINATION.md — ternary-entropy

> **Conservation Law Connection:** γ + η = C — entropy measures η

## Role in the Conservation Law

`ternary-entropy` provides the information-theoretic tools to **measure η directly**.
In the conservation law, η (overhead) is not just "wasted work" — it is the entropy
of the fleet's coordination signal. High entropy = high η = less productive capacity
available for γ.

The Shannon entropy H({p₊, p₀, p₋}) of a ternary distribution tells us the uncertainty
per agent. When all agents vote uniformly {-1, 0, +1}, H = log₂(3) ≈ 1.585 bits.
As the fleet converges to consensus, H → 0, and η → δ(n) (the irreducible CLT floor).

## delta-clt Verification Results

The delta-clt Monte Carlo shows η (normalized fleet imbalance) closely tracks δ(n)
for independent agents. This means:

- At n=50: η ≈ 0.121 (12.1% overhead), H decreasing
- At n=500: η ≈ 0.038 (3.8% overhead), H → 0

The correlated fleet simulation (30% shared bias) shows **slower entropy decay** —
η is 2-3× higher than independent prediction. This means correlated training data
(opensource models trained on similar corpora) inflates η measurably.

**Implication:** Use `ternary-entropy` sliding-window analysis on live fleet
signals to detect when η exceeds δ(n) predictions — it indicates hidden correlations.

## Cross-Repo Connections

### → ternary-types
Depends on `Trit` for its probability distributions. The type IS the sample space.

**Shared:** Both define what "ternary" means at different levels — type vs. distribution.
**Different:** `ternary-types` is a type system; this crate is a measurement system.

### → conservation-languages
`conservation-languages` benchmarks the conservation law across 9 programming languages.
Each language implementation has different entropy characteristics (allocation patterns,
memory overhead). `ternary-entropy` could be used to score language implementations
by their η contribution.

**Shared:** Both study η from different angles — one information-theoretic, one
implementation-level.
**Different:** `ternary-entropy` is Rust library code; `conservation-languages` is
cross-language benchmarking and analysis.

### → ternary-rhythm
`ternary-rhythm` studies temporal patterns in ternary signals. Combined with entropy
measures, they form a complete picture: entropy = how uncertain, rhythm = when it changes.
High entropy + low rhythm = noise. Low entropy + high rhythm = convergence.

**Shared:** Both analyze ternary signal properties over time.
**Different:** Entropy is distribution-focused; rhythm is temporal-pattern-focused.

## Fleet Position

```
┌──────────────────────────────────────────────┐
│  ternary-entropy — THE η MEASURER             │
│                                               │
│  Shannon H ← ternary-types (Trit)             │
│  Sliding window → detect η > δ(n) anomalies   │
│  KL divergence → measure fleet drift          │
│                                               │
│  Feeds into:                                  │
│  ├─ delta-clt (verification baseline)         │
│  ├─ conservation-languages (η per language)   │
│  └─ ternary-rhythm (temporal η patterns)      │
└──────────────────────────────────────────────┘
```

