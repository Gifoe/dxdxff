# N8F CANE-PATH-CP NEZ 80

## Analysis Status

This implementation is a **post-hoc 80-patient sensitivity analysis**, not the frozen primary old-90 cohort. The ten exclusions are unconfirmed review cases and are not represented as confirmed label errors.

## Method

The label contract is fixed: `NEZ=1`, `EZ=0`, and `sigmoid(final_nez_logit)=P(NEZ)`. The Step4B feature backbone produces a direct NEZ logit. Three bounded, label-blind residuals are added:

1. A clean-NEZ prototype residual, initialized from fit-only clean-NEZ embeddings.
2. A multi-seizure evidence residual from seizure-channel embeddings.
3. A six-feature offline ridge-VAR directed-influence proxy residual.

The raw cache is used only by the offline causal feature builder. Raw waveforms, labels, center IDs, true class counts, and oracle thresholds are not model-forward inputs.

## PATH Decision Rule

Each patient's final NEZ logits are standardized by the patient median and IQR. A 23-dimensional label- and center-blind summary is passed to a bounded `23 -> 16 -> 1` threshold head. A valid channel is predicted NEZ exactly when:

```text
standardized_nez_logit >= predicted_patient_threshold
```

No Top-K or cardinality estimate is used. Oracle thresholds are diagnostics and training targets only for outer-train inner-OOF records. The outer-test set is evaluated after the ranking model and PATH head are frozen.

## Nested Protocol

- Outer folds retain the canonical old-90 fold assignment after exact removal of the ten manifest subjects.
- Each outer-train set is cross-fitted with four patient-wise inner folds using seed 42.
- Model seeds 42, 43, and 44 are averaged with equal fixed weights for the formal P2 ensemble.
- Outer-test labels never select checkpoints, profiles, seeds, ensemble weights, or thresholds.
- P2 is predeclared as formal; P0/P1/P3/P4 are reporting baselines and ablations.

## Running

Configuration-only validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_step4d_cane_path_cp_nez_80.ps1 -DryRun
```

Formal execution requires the 28-dimensional feature cache, raw cache for one-time offline causal extraction, a passed feature-cache audit, and the exclusion manifest. The runner fails closed when required formal files or causal coverage audits are missing.

## Limitations

The cohort is post-hoc. Observed channel labels are not absolute pathological truth. Ridge-VAR features are directed-influence proxies, not validated causal identification or TFCCM. The implementation does not guarantee patient macro-F1 above 0.70 and requires confirmation on the frozen 90-patient primary cohort.
