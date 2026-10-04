# Omni-iEEG patient-level 5-fold CV: baseline vs A1-context plugin

## Protocol and scope

This is a patient-level interictal localization benchmark, not a reproduction of
the official pooled Omni Task-2 test metric.  The unit is a unique
patient-channel, after arithmetic mean aggregation of logits over all eligible
interictal EDF/session occurrences and all 30 fixed two-second segments in the
predefined 60-second window.  A patient is scored only if it contains at least
one pathological/SOZ channel and one normal channel.  The primary endpoint is
the unweighted mean of per-patient AUROC across five patient-held-out folds.

All training hyperparameters were locked before fold 1: seed 42, five patient
folds, 30 epochs, AdamW (learning rate 1e-4, weight decay 1e-3), gradient clip
1.0, and the fixed 0.5 threshold only for secondary classification diagnostics.
There was no validation split, early stopping, threshold selection, or
hyperparameter search.  The plugin differs from the baseline only by the
A1-style two-head patient-relative-z attention residual.  Both variants use the
same frozen manifest, records, 30 segments, labels, optimizer, initialization
policy, aggregation, and evaluation implementation.

The frozen OOF evaluation completed once after checkpoint hashes were locked.
The first evaluator process ended in a native Windows access violation before
any patient result was written.  The resumed evaluator reused only atomically
written patient-model predictions that were hash-bound to the frozen protocol,
checkpoint, and freeze file.  It did not change a model, threshold, cohort, or
metric.

## Cohort and fold audit

The official Task-2 recording filters retained 253 patients and 636 eligible
interictal EDFs.  Among patients, 124 were both-class, 89
pathological-only, 29 normal-only, and 11 had no valid label.  The latter three
groups were excluded before prediction because a within-patient binary AUROC is
undefined for them.  The final benchmark therefore contains **124 both-class
patients**.  There were no conflicting patient-channel labels.

The center composition is HUP 10 patients, Open-iEEG 103, and SourceSink 11.
Each fold contains 24 or 25 patients: two HUP patients per fold, 20 or 21
Open-iEEG patients per fold, and two or three SourceSink patients per fold.
Pathological fractions remain uneven, especially for the small SourceSink
center (fold 4: 0.399), which is an unavoidable small-center limitation rather
than a model-selected split.

Unlabelled good channels were observable context only: 2,339 occurrences in
the retained-record cache.  They contributed neither loss nor metrics.

## Primary OOF results

Values below are patient-equal means over all 124 OOF patients; the `sd` values
describe heterogeneity across patients, not uncertainty of the mean.

| Model | Patient AUROC mean (sd) | Patient AP mean (sd) | MRR | Top-1 | NDCG |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7622 (0.2230) | 0.4773 (0.3298) | 0.6075 | 0.5081 | 0.6707 |
| A1-context plugin | 0.7630 (0.2187) | 0.4727 (0.3212) | 0.6297 | 0.5323 | 0.6716 |

The observed paired primary effect is **plugin minus baseline AUROC = +0.00085**.
The patient-cluster bootstrap (10,000 draws, seed 42) 95% CI is
**[-0.02935, +0.03149]**.  Of patients, 46.77% improved, 6.45% were unchanged,
and 46.77% worsened.  The observed paired AP effect is -0.00451 (95% CI
[-0.04066, +0.03228]).

The fixed-threshold (0.5) diagnostics are not threshold-tuned benchmark
results.  Baseline versus plugin respectively: Macro-F1 0.5452 vs 0.5095,
pathological F1 0.1592 vs 0.0885, normal F1 0.9312 vs 0.9305, and balanced
accuracy 0.5736 vs 0.5350.  The plugin did not improve these diagnostics.

The secondary pooled patient-channel diagnostic is also unfavorable: baseline
AUROC/AP 0.7157/0.3727 and plugin 0.7071/0.3263 across 10,202 labeled units.
It is not the primary endpoint.

## Fold-level results

| Fold | n | Baseline AUROC | Plugin AUROC | Baseline AP | Plugin AP | Baseline Macro-F1@0.5 | Plugin Macro-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 0.8405 | 0.7545 | 0.5710 | 0.4924 | 0.6670 | 0.5751 |
| 2 | 24 | 0.7309 | 0.7355 | 0.4315 | 0.4869 | 0.4959 | 0.5076 |
| 3 | 25 | 0.6367 | 0.7295 | 0.3517 | 0.4440 | 0.4610 | 0.4945 |
| 4 | 25 | 0.7634 | 0.7644 | 0.4760 | 0.4342 | 0.5243 | 0.4644 |
| 5 | 25 | 0.8381 | 0.8302 | 0.5542 | 0.5067 | 0.5758 | 0.5058 |

The apparent gains in folds 2 and 3 are offset by loss in fold 1 and do not
survive the patient-paired aggregate comparison.

## Center analysis

The plugin is not consistently beneficial by center.  Patient-equal AUROC
baseline -> plugin is HUP 0.8059 -> 0.7208, Open-iEEG 0.7643 -> 0.7744, and
SourceSink 0.7022 -> 0.6948.  The only positive center difference is the large
Open-iEEG cohort (+0.0101); it is counterbalanced by HUP and SourceSink.  These
small-center estimates are unstable (HUP n=10; SourceSink n=11), so they do not
establish a center-specific mechanism.

## Predeclared decision

The result is `INSUFFICIENT_PLUGIN_GAIN`: +0.00085 is below the predeclared
+0.01 minimum, well below the +0.02 meaningful and +0.03 strong regimes, and
its bootstrap interval includes materially negative as well as positive values.
The plugin AUROC (0.7630) does not reach 0.83.  This is a micro-improvement in
the primary ranking metric, not evidence for a substantial interictal
patient-context effect; AP and fixed-threshold diagnostics are worse.

The data formulation itself remains coherent with Dataset 1: both datasets ask
for patient-level ranking/localization of implanted pathological channels, with
Dataset 1 using ictal evidence (EZ versus NEZ) and this dataset using
interictal-only evidence (SOZ/pathological versus normal).  This experiment
does **not** support the narrower claim that the existing A1-style context
plugin materially compensates for the missing ictal dynamics.  Per the stop
rule, no ranking loss, event detector, HFO branch, calibration, larger context
model, or other follow-up mechanism was added.

## Artifact map

- `results/PROTOCOL_LOCK.json` and `results/MODEL_PROTOCOL_LOCK_V2.json`:
  immutable cohort/model protocol.
- `results/*COHORT*`, `results/PATIENT_CLASS_SUPPORT_ALL.csv`,
  `results/PATIENT_5FOLD_MANIFEST.csv`, and `results/FOLD_BALANCE_AUDIT.csv`:
  cohort and split audits.
- `results/BASELINE_TRAINING_AUDIT.csv` and
  `results/PLUGIN_TRAINING_AUDIT.csv`: all ten runs reached epoch 30 without
  test-fold label/prediction access during optimization.
- `results/OOF_SUMMARY.csv`, `results/FOLD_METRICS.csv`,
  `results/PAIRED_BOOTSTRAP.csv`, `results/CENTERWISE_OOF_METRICS.csv`, and
  `results/SECONDARY_POOLED_METRICS.csv`: compact non-identifying OOF results.

Patient/channel predictions, per-patient metrics, paired patient deltas,
checkpoints, raw recordings, caches, and runtime logs remain private and are
not included in this repository.
