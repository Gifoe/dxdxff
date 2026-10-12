# Ictal D2: jointly remove Source and ZDELTA, seed42

One user-requested combined deletion: original shared-only PR-MLP (8817
parameters), with Patient Z retained and all26 ZDELTA main/auxiliary columns
masked to zero after the original FIT-only preprocessing. Keep ABS, DELTA,
RATIO (62 active columns; 88D interface and LayerNorm unchanged).

Train this single configuration from scratch on all five original folds with
seed42+1009*fold, the exact original trainer, identical shared initialization,
patient-equal updates and validation-only checkpoint/threshold selection.
Source-free has no V/u/source bias and no source regularization. This is not R0,
not a warm start from a final checkpoint, and not a prediction ensemble.

Reuse the previous full feature/model/mask/source audit and the immutable
private scores of Full D2, w/o Source and w/o ZDELTA. All original inputs and
source hashes are checked. A new combined-path gate checks actual masked FIT/
validation inputs, shared initialization, source invariance, CUDA gradients
and exact interrupted training resume before scientific training.

Freeze all five new selections and the15 reused reference model bindings
before new TEST inference. Seal all20 score artifacts before full-cohort labels
are materialized for metrics. Each model contributes80 unique TEST patients,
7635 channels, under patient-equal aggregation. Paired bootstrap uses10000
unique-patient draws, seed42, fixed thresholds, **combination minus reference**.

The combination was requested after the earlier outer ablation results were
viewed. Its scientific status is **posthoc outer-informed exploratory repeated
testing**, not independent confirmation. There is no additional arm or tuning
within this run. Bootstrap intervals are unadjusted and conditional on seed42.

## Run on the original Windows server

The previous seven-arm code, original source and private artifacts must remain
available. Use the original training runtime for preparation/training/inference;
use the already admitted alternative analysis runtime only for verification:

```powershell
$code='E:\DRE-nips\new-pipeline\7-11\d2_no_source_no_zdelta_outer_seed42_v1\code\run_combo.py'
$py='C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe'
& $py $code prepare
& $py $code run
& 'E:\Anaconda\python.exe' $code verify
```

Verification does not train or run a model: it independently reads sealed
scores, checks sklearn metrics and all paired intervals. The analysis-only
runtime avoids the sklearn/NumPy native issue diagnosed in the previous study.
Old data, checkpoints and predictions are never overwritten or retrained.
No installed skill is used. Public artifacts are code and aggregate results/
audits only; patient records, scientific caches, checkpoints and logs stay private.
