# D2 ictal frozen outer — seed42

Completed user-authorized D2-only outer evaluation: **80 unique patients / 7,635
channels**, using five historical checkpoints, FIT-only preprocessors, and fixed
validation thresholds. No retraining, threshold tuning, or raw EEG extraction.

Patient-equal: **AUROC 0.730445, AP 0.533483, Macro-F1 0.620403,
EZ-F1 0.410909, MRR 0.737216, Top1 0.650000**.

Read [FINAL_REPORT.md](FINAL_REPORT.md) for all 11 metrics, patient bootstrap 95%
intervals, fold/source results, pooled diagnostics, and limitations. This is a
posthoc exploratory repeated-test amendment, not a newly passed original gate.

## Server commands

Run from the original source layout on the original Windows server. Existing
private inputs are pinned in `PROTOCOL_AMENDMENT.json`; do not substitute exports
or refit models. The commands below describe the completed execution; do not rerun
completed scientific evaluation as another test.

```powershell
$Py='C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe'
$Experiment='E:\DRE-nips\new-pipeline\7-11\d2_ictal_outer_seed42_v1'
& $Py "$Experiment\tests\test_contract.py"
& $Py "$Experiment\code\evaluate_outer.py" prepare
& $Py "$Experiment\code\evaluate_outer.py" evaluate
& $Py "$Experiment\code\verify_aggregates.py"
```

Private runtime: `C:\d2_ictal_outer_seed42_runtime`. Public aggregate allowlist:
the nine CSV/JSON files in `results/`. None contains patient identifiers or
channel predictions. Original D2 artifacts are never overwritten.
