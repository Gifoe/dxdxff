# D2 ictal complete ablation, seed42

Seven prescribed configurations, original 80-patient / 7635-channel fixed
five-fold outer protocol. Reuse Full D2 and w/o Patient Z frozen predictions;
train exactly five new configurations on all five folds. This is an
**exploratory repeated-outer ablation**, not architecture search, independent
external validation or unseen-center generalization.

| Configuration | Patient Z | Source | ABS | DELTA | ZDELTA | RATIO | Active columns | Parameters |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| Full D2 | Yes | Yes | Yes | Yes | Yes | Yes | 88 | 9221 |
| w/o Patient Z | No | Yes | Yes | Yes | Yes | Yes | 88 | 9221 |
| w/o Source | Yes | No | Yes | Yes | Yes | Yes | 88 | 8817 |
| ABS Only | Yes | Yes | Yes | No | No | No | 18 | 9221 |
| w/o DELTA | Yes | Yes | Yes | No | Yes | Yes | 62 | 9221 |
| w/o ZDELTA | Yes | Yes | Yes | Yes | No | Yes | 62 | 9221 |
| w/o RATIO | Yes | Yes | Yes | Yes | Yes | No | 70 | 9221 |

`DELTA` denotes the pre-ictal-reference deviation view, not delta-band power.
All feature ablations preserve an 88D interface and the original LayerNorm and
backbone. Mask excluded columns to exact zero **after** complete original
preprocessing, for FIT, validation and TEST. No feature deduplication.

Before training, the runner validates feature order/generation code, duplicate
and constant columns, immutable reference files, original model replay,
FIT-only normalization, masks, shared initialization and cohort partitions.
Engineering tests additionally check source-free source invariance, gradients
and exact epoch-boundary CUDA resume for full/masked/source-free constructors.

The unmodified original trainer is called; w/o Source supplies only the original
shared network through a constructor adapter. It has no V, u or source bias,
and no source penalty. Historical rank-zero R0 retains source biases and is not
used. Every new arm/fold selects its own checkpoint and threshold on validation,
under the original selection rules. All 35 selected model/preprocessor/mask/
threshold bindings are frozen before new TEST inference; all prediction files
seal before full-cohort labels are loaded for new metrics.

## Execution on the original Windows server

Deploy this directory under the original source root. Use the original runtime:

```powershell
$py = 'C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe'
$code = 'E:\DRE-nips\new-pipeline\7-11\d2_ictal_ablation_seed42_v1'
& $py "$code\code\run_study.py" audit
& $py "$code\code\run_study.py" prepare
& $py "$code\tests\engineering.py"
& $py "$code\code\run_study.py" run
& $py "$code\code\verify_completed.py"
```

The feature audit must pass before preparation, and engineering tests must pass
before scientific training. Training retains original epoch checkpoints and can
resume without losing completed work. Completed scientific outer evaluation is
not rerunnable. Verification reads sealed scores without model replay or tuning.

Final public results include 7 overall rows, 35 fold rows, 28 center rows and
paired bootstrap comparisons. All 11 metrics are patient-equal; all 66 paired
effects use **Full minus ablated**, 10000 unique-patient resamples, seed42.
Intervals are descriptive, unadjusted for multiple exploratory comparisons.

Only code and aggregate results/audits are public. Raw EEG, feature archives,
patient/channel identifiers, checkpoints, private predictions and runtime logs
remain on the server. No installed skill was used for this task.
