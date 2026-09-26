# Retrospective B0/BCR ranking headroom audit

This is an oracle-only analysis on historical outer outcomes, not a deployable model or a trained distillation student. Alignment: 80 patients, 7,635 exact patient/fold/channel keys, no label mismatch.

- B0 rank-space patient-threshold oracle Macro-F1: `0.702602`.
- Best **global-lambda oracle**: `lambda=0.8`, Macro-F1 `0.713157` (delta `+0.010555`). Lambda was selected using outer labels and is not deployable.
- Best **patient-lambda oracle**: Macro-F1 `0.739260`. Each patient's lambda was selected using that patient's labels; it is a stronger non-deployable bound.
- Global mix ranking vs B0: EZ-AUPRC `0.559563` vs `0.531996`; EZ-MRR `0.779238` vs `0.733031`; Top-1 EZ `0.7000` vs `0.6500`.
- Predeclared ceiling condition (>=0.72 or >=+0.015) met: `False`; ranking consistency (AUPRC and MRR higher, Top-1 nonnegative) met: `True`.

`RANK_DISTILLATION_WORTHWHILE = NO` under the locked diagnostic rule. Even YES would justify only a future separately preregistered distillation test, not a deployable 0.70 claim. The boundary diagnostics restrict to the fixed hardest 10/20/30% by distance to the frozen global B0 logit threshold; rows exclude subsets lacking either class and report eligible counts.
