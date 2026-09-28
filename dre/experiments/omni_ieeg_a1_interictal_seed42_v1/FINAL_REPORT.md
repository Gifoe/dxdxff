# Omni-iEEG × A1 interictal, seed 42 — final report

## Outcome

The frozen A1-Omni model completed one official-test pass. The primary
patient-equal SOZ/EZ average precision (AP) was **0.3102** (10,000-patient
cluster-bootstrap 95% CI **0.2585–0.3650**). Pooled channel AP was **0.2279**
(95% CI **0.1782–0.2841**), compared with 0.0829 SOZ prevalence. This is a
ranking signal, but **not** evidence of superiority over the published Omni
models: those model labels/protocols are not identical to strict SOZ-only
labels here, and no matched baseline was trained in this run.

| Test metric | Value | Patient-cluster 95% CI, if computed |
| --- | ---: | ---: |
| Patient-equal mean AP (primary) | 0.3102 | 0.2585–0.3650 |
| Patient-median AP [Q1, Q3] | 0.2425 [0.0968, 0.4806] | — |
| Pooled SOZ AP | 0.2279 | 0.1782–0.2841 |
| Pooled AUROC | 0.7096 | 0.6666–0.7504 |
| Patient-equal MRR | 0.4335 | 0.3528–0.5155 |
| Patient-equal Top1-SOZ | 0.3085 | 0.2128–0.4043 |
| Patient-equal NDCG | 0.5399 | — |
| Macro-F1, fixed threshold 0.5 | 0.6030 | 0.5744–0.6305 |
| SOZ/EZ-F1, fixed threshold 0.5 | 0.2862 | — |
| Balanced accuracy, fixed threshold 0.5 | 0.6250 | — |
| Predicted SOZ fraction | 0.1188 | — |

The test contains **94 patients**, **102 labeled EDFs**, **9,215 unique
patient-channels** (764 SOZ), and 10,099 labeled EDF-channel records before
cross-record aggregation. Six test patients have no SOZ-labeled channel; by
the prespecified convention, their patient AP, MRR and Top1 are zero, and
their per-patient AUROC is undefined.

## Source and data contract

- Official Omni code: `57c22a75a59b5c3a98006806ad42000f6a3fa5b6`.
- Public data revision: `73b9c5180a57828ab2a83c040e7e9d112e77b2cc`.
- Native-signal cache: `F:\Omni-iEEG\signal_cache` on the original Windows
  server, schema `omni-ieeg-native-digital-v1`; 1,049 HDF5 EDF equivalents,
  3,228 byte-identical sidecars, 94,660,445,673 cache bytes. The native
  signal and sidecar validation passed before this experiment.
- Split: exactly the official `derivatives/datasplit/final_split.csv` patient
  membership, with train/test disjoint. Primary Task-2 inclusion:
  train/test, `frequency > 900`, `interictal=True`, `length >= 62`, and
  `dataset != Multicenter`; within each EDF, `good=1` and known `soz` 0/1.
  No patient was moved between official train and test.
- The filtered official universe is **151 train patients / 399 EDFs** and
  **102 test patients / 237 EDFs**. All 385 Zurich EDFs (250 train, 135 test)
  have no `good=1` channel with known SOZ 0/1; their annotations are `-1`.
  Because the protocol explicitly excludes unknown SOZ labels, supervised
  training/evaluation can use only **139 train patients / 149 EDFs** and
  **94 test patients / 102 EDFs**. This is a material coverage limitation,
  not a discretionary center exclusion. Resection and surgical outcome were
  never substituted for missing SOZ labels.
- Clinical positive class is **EZ = SOZ = 1**; internal historical A1 logits
  are NEZ-positive and are inverted for SOZ scoring. The 20-train-patient
  sign audit passed. Neither resection nor outcome contributes to labels.

## Input, model, and training

The training extraction uses the official maximum-five random 60-second
clip count/start algorithm with a deterministic per-EDF seed. A1 channel
attention needs synchronous channels, so the same starts are shared among
channels of each EDF; the official independent-channel model samples each
channel separately. At test, **all** complete nonoverlapping 60-second
segments after the one-second edge exclusion are aggregated. No ictal data
or seizure onset is used. Cached native signals are converted to microvolts,
notched at 60 Hz, anti-aliased to 300 Hz, then divided into 59 overlapping
2-second windows at 1-second stride. A real-signal numerical replay matched
the official-style MNE preprocessing path within `1.21e-13` microvolts.

The nine descriptors are the exact historical A1 source implementations:
log-bandpower delta/theta/beta/low-gamma/high-gamma, RMS, variance, line
length per second, and spectral entropy. The interictal reference is the
within-segment temporal median with `1.4826 × MAD + 1e-5`; the four 9-D views
are absolute, median difference, robust standardized difference, and the
historical safe log-ratio. Thus input remains 36-D. Five-train-patient
feature-distribution/finite-value checks passed.

The **27,713-parameter** original A1 backbone was trained from scratch,
without old A1 weights, new architecture, auxiliary branch, or dataset ID
as input. It retains temporal and cross-record aggregation, patient-relative
z, channel attention and nonlinear classifier. The objective is the exact
patient-equal weighted BCE with EZ:NEZ weights 2:1, with patient-equal
sampling; optimizer AdamW, LR `1e-4`, weight decay `1e-3`, patient batch 2.

Only official-train patients were used for normalization and model selection.
The deterministic seed-42 inner split was 111 train / 28 validation patients,
stratified by dataset and SOZ presence. The maximum of 30 inner epochs was
selected by patient-equal SOZ AP; epoch **30** was selected (inner AP
**0.3564**). The model was reinitialized and refit on all 139 labeled train
patients for exactly 30 epochs. The checkpoint, normalizer, protocol and
train/validation split hashes were frozen in Git commit `78442c0` **before
official-test model scoring**. The 0.5 classification threshold was locked.

## Dataset-stratified test outcome

| Dataset | Patients | Channels | SOZ prevalence | Pooled AP | AUROC | Patient-equal AP | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HUP | 6 | 762 | 0.0669 | 0.1509 | 0.7250 | 0.2491 | 0.4500 | 0.3333 |
| Open-iEEG | 74 | 7,475 | 0.0788 | 0.2516 | 0.7272 | 0.3028 | 0.4121 | 0.2838 |
| SourceSink | 14 | 978 | 0.1268 | 0.1632 | 0.6221 | 0.3756 | 0.5394 | 0.4286 |
| Zurich | 0 labeled | 0 | — | — | — | — | — | — |

HUP and SourceSink pooled AP are lower than Open-iEEG, while their sample
sizes/prevalences differ. The experiment does not isolate a causal center
effect. Zurich cannot be assessed under the strict SOZ-only target.

## Interpretation and integrity

This run shows that the unchanged A1 backbone plus patient-equal strategy can
learn a **nontrivial interictal SOZ-ranking signal** from the labeled portion
of Omni-iEEG. It does **not** establish cross-dataset dominance, a published
baseline improvement, or validation on the full official filtered cohort.
The most important limitations are the all-unknown Zurich SOZ labels, one
seed, absence of a strict-label matched baseline, and the necessary
shared-channel training clip synchronization. Omni-iEEG is suitable as an
additional, explicitly qualified validation setting; it is not yet a sound
headline large-scale comparative benchmark for this model without resolving
label coverage and running a matched comparator.

The model was frozen before the official-test inference; the test was scored
once and **not used for tuning**. Ten thousand patient-cluster bootstrap
draws used seed 42. Independent output validation recomputed pooled AP,
AUROC and balanced accuracy; confirmed 94 disjoint test patients, 9,215
unique patient-channels, the 111/28 train-only split, and matching frozen
checkpoint hashes (`outputs/VALIDATION.json`: `pass=true`). No raw waveform,
feature cache, checkpoint, optimizer state, or runtime log is committed.

Result files: `outputs/PRIMARY_METRICS.json`, `PATIENT_LEVEL_METRICS.csv`,
`DATASET_STRATIFIED_METRICS.csv`, `PATIENT_CLUSTER_BOOTSTRAP.csv`,
`CHANNEL_PREDICTIONS.csv`, `CHECKPOINT_SELECTION.csv`, `TRAINING_AUDIT.json`,
`TEST_SCORE_FREEZE_AUDIT.json`, and `TEST_ACCESS_AUDIT.json`.
