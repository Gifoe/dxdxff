# A0 + PR-Spline, seed42 — completed development study

Terminal: `PR_SPLINE_DEVELOPMENT_GATE_FAILED`. B3 AUROC 0.710482 versus
exact A0 0.710667, scaling 0.710654 and matched residual 0.711366. Only 1/5
folds improves. No new Outer TEST, rescue model or feature extraction.
See [FINAL_REPORT.md](FINAL_REPORT.md) for all metrics and 15 research answers.

Exactly B0 frozen source-free A0 plus independently trained B1 positive
feature scaling, B2 width2 generic input residual and B3 bounded monotonic
feature curves. No raw or supplementary feature extraction, alternate
normalizer, source-conditioned model, extra seed or hyperparameter sweep.

The original88D interface remains.56 representative features receive one
curve each;16 auxiliary duplicates share their corresponding main curve;
16 zero columns are untouched by B1/B3. B2 can alter zero/duplicate values
through its prescribed generic residual: masking it would change that control.

All new backbones are initialized through the exact original constructor
before extra modules. The unchanged original trainer is called via a temporary
constructor adapter, source penalty0. B0 is replayed rather than retrained.
Knots fit legal FIT patient/channel observations only, with equal patient mass.
Original Patient Z, imputer/scaler, objective, patient order, epoch stopping
and0.005-grid checkpoint/threshold selection are preserved.

```text
python code/baseline_admission.py
python code/prepare.py
python tests/engineering.py
python code/train.py
python code/evaluate.py
python code/verify.py
```

Execution requires the hash-bound trusted server artifacts. Source/knot/model
and complete optimizer/RNG states remain private; only anonymous aggregate
results, code and hashes are delivered. See `FINAL_REPORT.md` for actual
results and `results/SCIENTIFIC_TERMINAL.json` for the registered decision.

The primary unit is the original65 equally weighted validation-fold appearances
from47 unique IDs.10,000 paired seed42 bootstrap draws resample identities,
retaining all their appearances. All intervals are descriptive, unadjusted
and development-selected, not independent clinical validation.

No installed skills are used. No new Outer TEST path or authorization exists,
regardless of the continuation gate outcome. Original checkpoints and banks
are never modified. Runtime byte hashes refer to original server artifacts;
delivery aggregate text may be Git line-ending normalized.
