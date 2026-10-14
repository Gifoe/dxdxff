# DELTA reconstruction audit

Seven named physiological families produce exactly fourteen varying coordinates: feature-major seizure mean and population SD. The reference is the original float32 mean of **all genuine center < 0 windows**, not CoTAR's [-10,0) subset. Window-level differences are averaged over all genuine source windows; only genuinely present seizures enter the across-seizure statistics. Single-seizure SD is zero. No no-pre-reference fallback was needed or allowed.

The original evidence helper's full contiguous layout and per-channel/1D float32 reductions are required for numerical replay. The initial vectorized reduction failed 3 of 106,890 archived coordinates; the identified arithmetic correction is documented in `STAGE_A_ENGINEERING_REPAIR.md`, with the initial failure and diagnostic retained. The final complete genuine-support E14 is bit-identical to all named archived coordinates: maximum absolute error 0 and zero failed coordinates. Tolerance was not relaxed.

R14 is separately measured from exact Raw slices using the pinned spectral/time-domain functions and the same reference/aggregation arithmetic. Historical E14 is used only for fidelity and E1. D1 and the frozen D1 pathway inside D2/D3 receive R14, never original E14. The helper's intermediate duplicated/zero coordinates are not additional predictors.

Clinical labels did not enter Stage A. Category C discrepancies persist because bandwidth differs; success is recovery under the registered rank and predictive gates, not complete signal equality. No new Outer prediction or evaluation is authorized.
