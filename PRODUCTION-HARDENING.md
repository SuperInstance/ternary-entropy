# Production Hardening — Round 4 (2026-07-11)

Tracking log for the production-readiness pass on branch
`production-round4-2026-07-11`. This file is added as the scaffold commit;
substantive fixes land in their own commits after this one.

## Scope

- Verify the build/test/lint infrastructure actually passes (not fake-green).
- Verify information-theoretic claims numerically against known-correct values.
- Fix real bugs found in source.
- Reconcile README claims with what the code actually does.
- Add test coverage for genuine, previously-untested branches of logic.

## Verified baseline before fixes

- `cargo build --all-features` — OK
- `cargo test --all-features` — 24 passed, 0 failed
- `cargo clippy --all-features -- -D warnings` — **FAILS**
- `cargo fmt --all -- --check` — **FAILS**

## Findings (to be addressed in subsequent commits)

1. CI is red: clippy (`needless_range_loop`) and `rustfmt --check` fail on the
   default branch.
2. `JointDistribution::new` validates the sum but not non-negativity, so it
   accepts distributions with negative entries — inconsistent with
   `TernaryDistribution::new`.
3. README states Jensen-Shannon divergence is "bounded by log₂(3)". The tight
   bound for JS (base 2) is 1 bit (log₂ 2), independent of alphabet size.
4. `JointDistribution::from_pairs(&[])` yields an all-zero (invalid) table; the
   degenerate behavior is undocumented.
5. Several real branches lack test coverage; entropy values are not pinned to
   known-correct numeric results.
