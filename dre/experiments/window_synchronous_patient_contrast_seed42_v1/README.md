# Window-Synchronous Patient Contrast Audit — Seed42

Fixed mechanism audit, not a new deep model or Outer TEST. See PROTOCOL_LOCK.json and HISTORICAL_OVERLAP_AUDIT.md. The trusted-server driver is `code/run.py --runtime PRIVATE_RUNTIME --source HISTORICAL_CODE_ROOT`.

The three matched 182D arms use baseline + own-channel temporal summaries, plus either 84 neutral coordinates (C0), true query-excluded synchronous context (C1), or matched stage-local shifted context (C2). No count/identity/center predictor. Genuine cache windows only, no raw/EDF access.

Execution first tests label-free achieved disruption. Only a passing control permits sixty fixed logistic probes inside original FIT patients. All numeric thresholds and predictions seal before held-out scoring. Original Development/Outer labels and predictions are not used. Repeat appearances are clustered by the 80 unique patient identities for bootstrap.

## Completed results

The original FIT-only audit completed60 probes,255 repeated appearances/79 unique identities (80 in label-free data construction). C0/C1/C2 AUROC:0.646775/0.714885/0.688805. All original inner gates passed; see `results/`.

The user then explicitly authorized Outer and chose15 final probes with fixed TRAIN/threshold-selection splits within each original FIT set. This is a separate extension in `OUTER_PROTOCOL_LOCK.json`, `outer_code/` and `outer_results/`; the original no-Outer protocol and inner results remain unchanged.

| Outer arm | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| C0 own temporal |0.672639|0.480328|0.507376|0.294877|0.660688|0.5375|
| C1 synchronous |0.711834|0.507647|0.574056|0.345849|0.685648|0.5875|
| C2 shifted |0.698389|0.506131|0.557412|0.329349|0.726676|0.6500|

Outer covers80 unique patients/7635 channels. C1-C0 AUROC+0.039195 [0.002149,0.077865]; C1-C2+0.013444 [-0.006018,0.034423]. Synchronization-specific Outer CI gate fails; combined terminal **PATIENT_CONTEXT_POSSIBLY_USEFUL_SYNCHRONIZATION_NOT_IDENTIFIED**. No TIDE training or rescue. See FINAL_REPORT and MECHANISM_AUDIT; Outer is exploratory/repeated, not blind confirmation.

Public artifacts contain only code, hashes and nonidentifying aggregates; all data, IDs, labels, models, predictions and logs stay private. Original source/repair hashes are preserved. The historical alignment is exact reused code, not an installed skill.

Outer trusted-server invocation: `outer_code/run.py --inner PRIVATE_INNER_RUNTIME --runtime NEW_PRIVATE_OUTER_RUNTIME`. All15 final models/thresholds seal before Outer inference and all scores before clinical y access; no original Development evaluation. Tests:13 original plus2 Outer extension synthetic tests. Stop after this fixed evaluation.
