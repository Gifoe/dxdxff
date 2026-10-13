# COVE: conditional-innovation view audit, seed42

Terminal: `COVE_DEVELOPMENT_GATE_FAILED`. Completed all15 new model-fold cells, four-arm five-fold development comparison,10,000-draw patient-ID bootstrap, frozen interventions and14-group attribution. **No new outer TEST evaluation.**

Primary report: [FINAL_REPORT.md](FINAL_REPORT.md). Feature-focused findings: [CONDITIONAL_INFORMATION_AUDIT.md](CONDITIONAL_INFORMATION_AUDIT.md), [group correction effects](results/GROUP_INTERACTION_EFFECTS.csv), [cache magnitudes](results/FEATURE_GEOMETRY.csv), [original Patient-Z floor incidence](results/PATIENT_Z_FLOOR_AUDIT.csv), and [paired results](results/PAIRED_BOOTSTRAP.csv).

Main execution order (trusted private dependencies required):

```powershell
python code/prepare.py --runtime D:\a0_cove_seed42_v1_runtime_r3
python code/train.py --runtime D:\a0_cove_seed42_v1_runtime_r3
python code/evaluate.py --runtime D:\a0_cove_seed42_v1_runtime_r3
python code/verify.py --runtime D:\a0_cove_seed42_v1_runtime_r3
```

The actual formal runtime is the numerically verified Python3.13.9/PyTorch2.11.0+cu128 environment. `tests/contract.py` runs inside preparation; it is not a mock-only deliverable. Initial Python3.12 preparation failures and all private partial artifacts were preserved. Runtime parity probes are in `verification/`; no unsafe OpenMP duplicate-library bypass was used.

Post-completion score-only attribution and label-free interpretive diagnostics:

```powershell
python verification/aggregate_attribution.py --runtime D:\a0_cove_seed42_v1_runtime_r3
python verification/feature_geometry.py --runtime D:\a0_cove_seed42_v1_runtime_r3
python verification/patient_floor_audit.py --runtime D:\a0_cove_seed42_v1_runtime_r3
```

These do not rerun models, read outer data, remove features or alter any selection. Main scientific code/test/protocol hashes are frozen before training. Nuisance fits use only legal FIT patients; full-FIT nuisance models alone apply to validation. Ridge coefficients/normalizers and individual identities/predictions remain private. Public artifacts are anonymous aggregate statistics and hashes. Original raw/source/checkpoints are not modified.

Result bytes preserve the server's Windows line endings (`results/* -text`) so audit SHA256s are directly checkable after publication. Source files use LF. No installed skill was used. No extra arm, automatic v2, feature-subset training or outer inference follows this failed gate.
