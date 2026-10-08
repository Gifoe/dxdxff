# E1 random-init patient-relative CNN, seed42

User request: finish E1 after the completed E3 experiment. Scope is the same source-only five-fold TRAIN/validation setup, not an outer benchmark. E0/E2/E4/E5 and independent EDF onset replay are not included.

E1 uses the supplied `IctalLocalization(use_pr=True, use_attention=False)` with **44,401 parameters**, no SSL pretraining and no pretrained encoder loading. Supervised seeds, optimizer, BCE, 30 epochs, patient-equal aggregation, checkpoint choice and per-fold validation threshold selection match E3. It reuses the exact approved E3 export: 255 seizures, all 80 patients and 7,635 channels, unchanged fold roles, one excluded padded seizure, no raw re-extraction.

The existing validated isolated NumPy1.26.4 environment is used from the start to avoid the original NumPy2.2.6 native metrics crashes. PyTorch2.8.0+cu128 is unchanged. This binary-runtime difference from E3 training is disclosed in the protocol; prior metric/sampling/input/forward parity and new E1 original-loop/resume tests are required.

`code/run_e1_validation.py` imports the **hash-locked** E3 helpers and original supplied model/metrics from the existing E3 amended-code folder. It never edits E3 source or checkpoints. Every complete E1 epoch stores model/optimizer/RNG state privately and rejects changed resume bindings. FIT+VAL loading rejects outer patients.

Server code: `E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e1_seed42_v1`.

Private runtime: `C:\ictal_onset_ssl_pr_e1_seed42_runtime\validation`.

Public artifacts contain only code and aggregate audits/results. E1/E3 individual validation rows, tensors, predictions, logs and checkpoint files remain private. Paired bootstrap resamples patient IDs, keeping all matched validation appearances; post-selection validation is not independent clinical confirmation. A computed E4/E5 development eligibility gate does not authorize running those controls.

## Completed results

**E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED**. Five-fold validation means: Macro-F1 **0.613295**, EZ-F1 **0.387922**, EZ-AUPRC **0.487951**, EZ-AUROC **0.706686**, MRR **0.709271**, Top1 **0.615385**.

Frozen E3 minus E1 Macro-F1 is **+0.003251**, paired patient-cluster 95% interval **[-0.018788, +0.026011]**. The predeclared SSL-increment development gate fails (3/5 positive folds; required gain +0.020 and 4/5 folds). No stable SSL benefit is established. No optional arms or outer tests were run.

See [FINAL_REPORT.md](FINAL_REPORT.md), [E1/E3 comparison](results/E1_E3_COMPARISON.csv), [paired bootstrap](results/paired_bootstrap_E3_minus_E1.csv), and [completion audit](audit/COMPLETION_AUDIT.json).
