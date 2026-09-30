# Engineering repair ledger

No repair changed the frozen data, model, loss, split, candidate grid, selection rule or gate.

1. Short valid waveforms were zero-padded to the frozen 60-s tensor length; the original A1 window mask still excludes every padded window.
2. Competitive statistics require all six group kernels to be valid at a time point, preventing an undefined top1-minus-top2 margin on shortened records.
3. The Windows scheduler wrapper separates Python stdout/stderr and trusts the Python exit code, avoiding false failure on warnings.
4. VLOO aggregation replaced repeated sklearn calls with algebraically equivalent vectorized confusion/ranking formulas. Random tied-score equivalence checks passed to 1e-12; single-class query cells retain sklearn's historical inferred-label semantics.
5. Inference diagnostics used a deterministic replay of the selected lightweight H1. Frozen full predictions matched within 1.79e-7 before any intervention.
