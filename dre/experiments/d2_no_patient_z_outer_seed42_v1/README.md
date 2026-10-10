# D2 without patient_z — ictal outer seed42

Five scratch-trained D2 folds, changing only patient-local cross-channel feature
z-standardization to FIT-only global imputer/scaler. All 88 physiological feature
columns, the 9,221-parameter rank4 model and historical training/selection rules
remain unchanged. This is not an absolute-only feature ablation.

80 unique held-out patients / 7,635 patient-channel units, patient-equal metrics:

| Variant | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| Original D2 PR frozen outer | 0.730445 | 0.533483 | 0.620403 | 0.410909 | 0.737216 | 0.650000 |
| D2 without patient_z, retrained | 0.660217 | 0.480013 | 0.493978 | 0.259845 | 0.685571 | 0.600000 |

Macro-F1 falls in 5/5 folds. Paired patient bootstrap delta -0.126425,
95% interval [-0.162872, -0.089640], 10,000 draws, seed42, no threshold refitting.
This is exploratory repeated-test evidence, not a fresh sealed confirmation.

See [FINAL_REPORT.md](FINAL_REPORT.md) for all metrics and qualifications, and
`results/` for full-precision absolute/paired intervals, fold/source aggregates,
validation-selected checkpoints/thresholds, training history and provenance audits.

Server source: `E:\DRE-nips\new-pipeline\7-11\d2_no_patient_z_outer_seed42_v1`.
Private runtime: `C:\d2_no_patient_z_outer_seed42_runtime`. The completed commands:

```powershell
$Py='C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe'
$Experiment='E:\DRE-nips\new-pipeline\7-11\d2_no_patient_z_outer_seed42_v1'
& $Py "$Experiment\tests\test_contract.py"
& $Py "$Experiment\code\run_ablation.py" prepare
& $Py "$Experiment\code\run_ablation.py" train
& $Py "$Experiment\code\run_ablation.py" freeze
& $Py "$Experiment\code\run_ablation.py" evaluate
& $Py "$Experiment\code\verify_completed.py"
```

Do not rerun completed scientific evaluation as another test. Preserve private
artifacts; publish only this source and the aggregate result allowlist.
