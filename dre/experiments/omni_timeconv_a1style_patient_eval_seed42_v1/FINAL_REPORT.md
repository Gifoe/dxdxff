# Official Omni TimeConv-CNN under an A1-style patient-equal protocol

**Terminal: `Case C` — within-patient ranking is high by the predeclared descriptive cutoff, but fixed-global-threshold Macro-F1 is low.** This is an additional A1-style patient/channel localization protocol applied to the frozen Omni-iEEG Task2 TEST predictions. It is **not** a replacement official Omni benchmark and, because this TEST cohort has been viewed previously, is an exploratory repeated-test analysis.

| Evaluation system | Unit | AUROC | Macro-F1 | AP | MRR | Top1 | NDCG |
|---|---|---:|---:|---:|---:|---:|---:|
| Official pooled | EDF-channel | 0.798768 | 0.659754 | 0.330301 | — | — | — |
| A1-style patient-equal | patient-channel | 0.817186 | 0.443078 | 0.748585 | 0.831977 | 0.772727 | 0.849479 |

## Direct answers

1. **Frozen pooled replay:** yes. AUROC `0.7987677712` is within the locked tolerance of `0.7987673466`.
2. **Official labeled EDF-channel units:** `8104` (7297 normal, 807 pathological).
3. **Unique patient-channel units after fixed EDF mean:** `5055`.
4. **TEST patients:** `96`.
5. **Patients with both classes:** `52`; single-class patients: `44`.
6. **Patient-equal AUROC:** `0.817186`.
7. **Patient AUROC distribution:** SD `0.213521`, median `0.896558`, IQR `[0.746162, 0.983213]` across the `52` estimable patients.
8. **Patient-equal Macro-F1 at 0.5:** `0.443078`. This averages every patient, including single-class patients, with `labels=[0,1]` and `zero_division=0`.
9. **Leakage-free A1-style validation threshold available?** No. The only recovered full-record TRAIN artifact contains in-sample predictions from this model's fit population, not a patient-held-out validation artifact; selecting a threshold there would be optimistic reuse. No threshold was fabricated.
10. **Thresholded TEST Macro-F1 from a train/validation selection:** unavailable by design.
11. **Pooled minus patient-equal AUROC:** `-0.018418`. This is descriptive, not causal decomposition.
12. **Center variation:** see table below; no center or patient was removed based on results.
13. **Zurich estimable?** No: all Zurich test patients are single-class under this frozen label set, so patient AUROC is explicitly undefined rather than set to 0.5.
14. **Where is TimeConv stronger?** The observed comparison is stated by `Case C` above; it distinguishes global pooled discrimination from within-patient localization without changing either metric definition.
15. **Implication for unified modeling:** this measurement alone cannot establish a mechanism. It only indicates whether retaining patient-relative channel structure is a reasonable hypothesis to test prospectively; it does not validate a modified model.

## Patient AUROC by center

| Center | Patients | AUROC-estimable patients | Patient-equal AUROC |
|---|---:|---:|---:|
| HUP | 6 | 4 | 0.573874 |
| Open-iEEG | 72 | 44 | 0.847884 |
| SourceSink | 14 | 4 | 0.722819 |
| Zurich | 4 | 0 | not_estimable |

## Provenance and safeguards

- The CNN was not loaded and no neural inference was run. Scores were recomputed only from the frozen segment-logit cache.
- The cache hard-bound the official checkpoint `442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852` and source `c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95`. Its cache digest was `3f5cf9a9111fad6c5843eeaba43108057df961dd6a918d83fc40cc34bb5053ad` and it is linked by earlier replay provenance to frozen channel-prediction fingerprint `d8a500a578c613d177e6513c8692040ab25d5fe0da64fba22631b7b9ae4eaa14`.
- The score is `mean_t(1 - sigmoid(normal_logit_t))`; this is sigmoid per segment followed by averaging, never `sigmoid(mean(logit))`.
- Official raw label `1=normal, 0=SOZ/pathological, -1=excluded` was converted once during frozen cache construction to `pathological_labels = 1 - official_label`; this evaluation used those frozen values unchanged.
- Identifying patient/channel rows are retained only in the private runtime directory. Git-tracked files are aggregate-only.
