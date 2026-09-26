# Implementation audit

- Locked protocol SHA-256: `55fa533704fa0452b1f20ab409a818fb8e96262067bd0bac264e2fceb480b75c`; A1 source reproduced exactly over 5 folds × 30 epochs × 13 validation patients.
- A/B/C: 900 trained 15-epoch residual heads, matched initialization and parameter counts within each control/candidate pair; A1 frozen throughout.
- A1 trajectory projection was FIT-only; deterministic ordered/shuffled controls had equal 32-column head input after zero-padding unused PCA columns.
- B recruitment standardization and magnitude-control thresholds were fitted on FIT only; C reused the same rank evidence and retained one-seizure patients.
- All 150 fold/epoch cached normalized inputs, centers and validity masks were independently checked bitwise equal within fold; the fold-level recruitment feature cache is numerically exact.
- D: original A1 D0 reused, 3 matched ablations × 5 folds × 30 epochs trained from the exact fold initialization, with full 36-D architecture and view masking after the original FIT normalizer.
- Private intermediate cache: 150 cells; schema 36-D input, 32-D window/temporal embeddings, 64-D patient/contextual embeddings. Private cache bytes: 73059965888.
- Engineering repair: NumPy native access violation during quantile of short strided seizure arrays was replaced by a pure-Python linear-interpolation quantile with identical mathematical semantics; all completed cells and checkpoints were preserved and resumed.
- One unreadable private representation cache cell was quarantined and rebuilt from its exact frozen A1 checkpoint; the six existing probe heads and validation grids were preserved. The subsequent 150-cell input invariance scan passed.
- Patient-level representations, IDs, labels, scores, failure rows, model checkpoints and logs remain in the private server runtime. Public outputs are aggregate only.
- No outer-test loader was constructed and no outer predictions/metrics were evaluated. The underlying historical cache constructor indexes cohort metadata; no outer labels or outcomes were used for training, selection, diagnostics or gates.
