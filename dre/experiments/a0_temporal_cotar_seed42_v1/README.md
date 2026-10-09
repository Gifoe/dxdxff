# Frozen A0 + Temporal CoTAR, seed42

Completed: ten T1/T2 server runs and10,000 patient-ID cluster bootstrap draws.
Development gate **failed**; no outer evaluation performed.

| Arm | Patient Macro-F1 | EZ-F1 | EZ-AUPRC | EZ-AUROC |
|---|---:|---:|---:|---:|
| T0 frozen A0 | 0.638080 | 0.431198 | 0.518235 | 0.710667 |
| T1 local control | 0.641199 | 0.439072 | 0.522287 | 0.708445 |
| T2 CoTAR | 0.640770 | 0.439989 | 0.523259 | 0.711127 |

T2-T0 F1+0.002690 (95% CI[-0.003441,+0.009001]); T2-T1 -0.000429
(CI[-0.004416,+0.003116]). Target+0.03-0.04 not reached. Terminal
`TEMPORAL_FEATURES_USEFUL_COTAR_NOT_SUPPORTED` is the prompt's numerical branch,
not proof of clinical usefulness. See [FINAL_REPORT.md](FINAL_REPORT.md),
[results/](results/) and [audit/](audit/) for complete metrics and admission evidence.

Requested arms: **T1 and T2 only**, five original development folds each. T0 is
frozen A0 inference; no A0 student/teacher retraining, raw input, SSL or T3.
No new outer-test evaluation is authorized, even if development gates pass.

Read [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [PROTOCOL_LOCK.json](PROTOCOL_LOCK.json),
[TEMPORAL_DATA_AUDIT.md](TEMPORAL_DATA_AUDIT.md) and
[IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for source and scientific scope.
The existing all256 feature recordings,80 patients and7,635 channels are retained.

Private server execution order:

1. `code/audit_data.py`: original hash/identity/time/genuine-window/coverage gate.
2. `code/prepare.py`: freeze/replay A0; verify20 existing OOF teachers and generate
   deterministic eval logits; seal FIT/VAL and FIT-only temporal normalization.
3. `code/smoke.py`: synthetic tests and real FIT-only parity/gradient checks.
4. `code/run.py`: ten matched T1/T2 runs with eligible epoch0 fallback.
5. `code/finalize.py`: hash/metric checks,10,000 paired patient-ID cluster draws,
   fixed-T0-threshold, decision-change and residual diagnostics.

Each command has `--help`; private paths and caches are not bundled. The baseline
helper import requires the exact sibling PR-UAS source SHA, checked on import.
This directory disables Git newline conversion to preserve byte-level source
hashes recorded during the Windows server execution. Do not normalize the locked
source or protocol files before replaying their provenance gates.
Official parity tests require `TECH_LAYER_SOURCE` pointing to the pinned TeCh
layer file; its SHA is checked before importing it. Historical teacher provenance
and private feature export are mandatory, not replaceable arbitrary caches.

Only compact aggregate findings and implementation source belong here. All
temporal arrays, original caches, identifiers, individual predictions, A0 logits,
scalers, checkpoints and runtime logs remain on the trusted server. The public
result scope is validation-selected development (65 patient-fold appearances /
47 unique patients), not independent outer-test confirmation.
