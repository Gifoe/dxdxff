# Omni-PR mechanism audit — seed42

Frozen TimeConv and original 35-parameter PR references versus two matched new 35-parameter controls: globally standardized absolute embeddings and label-blind within-patient shuffled PR embeddings. No new CNN training or raw extraction.

## Completed result

Scientific terminal: `MECHANISM_INCONCLUSIVE`. Integrity passed, all 30 formal inner runs and 10 refits completed, and all four arms evaluated the same 124 patients/10,202 labeled canonical channels. Historical C0 and PR were reproduced exactly without retraining or rethresholding.

| Arm | Patient AUROC | Patient AP | Macro-F1, frozen FIT threshold | Pathological F1, frozen FIT threshold |
|---|---:|---:|---:|---:|
| C0 | 0.762163 | 0.477255 | 0.610352 | 0.313503 |
| Historical C1-PR | 0.811823 | 0.523527 | 0.655819 | 0.392346 |
| New C1-ABS | 0.813627 | 0.523435 | 0.626587 | 0.349214 |
| New C1-SHUFFLE | 0.753240 | 0.467298 | 0.603600 | 0.299685 |

PR-minus-ABS AUROC is -0.001804, exploratory paired-patient 95% CI [-0.012650, +0.008456], positive in 2/5 folds: the incremental patient-normalization gate fails. ABS alone recovers the historical ranking gain in point estimate, but the interval does not establish the predeclared +/-0.010 descriptive comparability rule, so `LINEAR_RECALIBRATION_SUFFICIENT` is not assigned.

The alignment gate passes: PR-minus-trained-SHUFFLE AUROC +0.058583, CI [+0.027768, +0.089873]. This supports using correctly aligned embeddings under the tested control, not a causal claim isolating a particular normalization mechanism. ABS-minus-C0 AUROC is +0.051464, CI [+0.020823, +0.082478].

Classification is a different result: PR-minus-ABS FIT-threshold Macro-F1 +0.029232, CI [+0.014365, +0.045284]. Removing patient mean at frozen PR inference preserves every ranking metric exactly, but changes Macro-F1 to 0.451420 at the original threshold. Do not conflate ranking preservation with thresholded classification preservation.

The simplest supported **ranking model class** remains a 35-parameter aligned linear correction of frozen CNN embeddings. Globally standardized ABS is a reasonable simpler normalization description, not a proved equivalent or superior model. Historical PR retains an observed classification advantage; it cannot yet be called necessary for the ~0.812 AUROC. No extra model or threshold was selected after these outcomes.

Read `PROTOCOL_LOCK.json`, `SOURCE_AUDIT.md` and `IMPLEMENTATION_AUDIT.md` before running. This is a repeated-cohort exploratory, patient-equal mechanism control, not a new blind pooled Task-2 benchmark. The private runtime and original pinned dependencies are necessary; no clinical data or checkpoint tensors are distributed here.

Execution on the original server, in order:

```powershell
python code/prepare.py
python tests/engineering.py
python code/train_controls.py
python code/evaluate.py
python tests/completion_verify.py
```

Only run preparation once for a new runtime. Formal training can resume from exact bound checkpoints without changing inputs or protocol. Evaluation requires the all-controls pre-outer freeze; completed private predictions cannot be replaced. Fetch only outputs validated by `COMPLETION_VERIFICATION.json`.

`FINAL_REPORT.md` contains the four-arm ranking/classification results, both frozen threshold policies, five paired patient contrasts, source-stratified effects, scale-only/inference-shuffle diagnostics, registered gates and 26 scientific answers. `audit/` contains compact integrity evidence; `results/` contains non-identifying aggregates only.
