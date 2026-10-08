# A0 + Patient-Local Label Reliability (PLLR), seed42

Training-only feature/label compatibility weighting of the exact historical
88D/8,817-parameter PR-MLP. No verified clinical label-error probabilities,
new inference inputs, soft targets, teachers or extra classifier are introduced.

Arms: original hard-label A0, within-patient/class permuted identical weights B1,
and feature-corresponding frozen weights B2. Same initialization, data-order
policy, optimizer and checkpoint/threshold rules in all arms. Original80-patient
cohort/folds, not raw E1/E3 seizure exclusions.

## Completed outcome

All 15 matched runs, FIT feasibility checks and 10,000-draw patient-ID cluster
bootstrap completed on the trusted server. The development gate **failed**:

| Arm | Patient Macro-F1 | EZ-F1 | EZ-AUPRC | EZ-AUROC |
|---|---:|---:|---:|---:|
| A0 | .638080 | .431198 | .518235 | .710667 |
| B1 permuted | .637447 | .431203 | .520138 | .709253 |
| B2 PLLR | .638224 | .423335 | .515109 | .714029 |

B2−A0 Macro-F1 = +.000144, 95% paired cluster CI [−.011930, .011775].
Only 2/5 folds improve; EZ sensitivity declines. The method is feasible but has
no convincing predictive benefit. Stop with `PLLR_POSITIVE_BELOW_TARGET`
(numerical positivity only), without new outer-test evaluation or retuning.
These are 65 validation appearances / 47 unique patients, not independent test
results. Historical outer B0 .616167 is a separate reproduction scope.

See [FINAL_REPORT.md](FINAL_REPORT.md) for all metrics, intervals, center/fold
results, operating-point diagnostics, gate decisions and execution limitations.

## Reproduction order

Execution order on the trusted server:

1. Hash-check the original validated export and freshly replay all historical
   B0 checkpoints using the immutable PR-UAS `runtime_admission.py` in a **new**
   private runtime (do not overwrite the old task's audits).
2. `code/prepare.py`: seal per-fold FIT/VAL banks, fit PCA8 and compute FIT-only
   feasibility, 100-resample prototype stability and frozen B1/B2 weights.
3. Only on feasibility PASS, `code/smoke.py`: synthetic tests and actual FIT-only
   original A0/all-one-weight training parity and weighted-gradient checks.
4. `code/run.py`: freshly train/reproduce five A0 folds first, then B1 and B2.
   Reject any drift in A0 selections/identities or prior matched development F1.
5. `code/finalize.py`: independent completeness verification, paired 10,000-draw
   patient-ID cluster bootstrap and decision/ranking/center diagnostics.

Only compact aggregates belong in `audit/` and `results/`. All features, PCA
states, prototype/weight identities, scalers, checkpoints, private predictions
and logs remain on the server. A feasibility failure stops without any formal
model training. A development-positive result still requires separately locked
authorization for any later outer evaluation; no new outer test is run here.

See SOURCE_AUDIT.md, PROTOCOL_LOCK.json and IMPLEMENTATION_AUDIT.md for source,
sealing, runtime and initialization caveats. FINAL_REPORT.md and
FEASIBILITY_REPORT.md contain actual execution outcomes, not synthetic results.
