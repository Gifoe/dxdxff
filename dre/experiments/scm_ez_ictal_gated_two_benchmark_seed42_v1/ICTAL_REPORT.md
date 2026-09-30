# SCM-EZ Ictal development report

This is validation-only historical fixed-query VLOO development. No outer test or Omni data was accessed.

| Model | AUROC | AP | Macro-F1 | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical A1 | 0.746382 | 0.576743 | 0.620810 | 0.740038 | 0.654771 |
| SCM-EZ Full | 0.497126 | 0.339976 | 0.438187 | 0.462141 | 0.362296 |

SCM-EZ is not competitive. Relative to A1, the deltas are:

- AUROC: **-0.249256**
- AP: **-0.236767**
- Macro-F1: **-0.182623**
- MRR: **-0.277897**
- Top1: **-0.292475**

## Fold consistency

| Fold | SCM AUROC | A1 AUROC | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.324159 | 0.755657 | -0.431498 |
| 2 | 0.445708 | 0.767127 | -0.321420 |
| 3 | 0.696682 | 0.690787 | +0.005895 |
| 4 | 0.656519 | 0.781496 | -0.124977 |
| 5 | 0.366524 | 0.720280 | -0.353756 |

Only 1/5 folds beats A1. The variation from 0.324 to 0.697 is severe and rules out a stable gain.

## Gate

All required checks failed:

- AUROC >= 0.751382: FAIL
- AP >= 0.576743: FAIL
- positive folds >= 3: FAIL
- MRR >= 0.730: FAIL
- Macro-F1 >= 0.606: FAIL

The selected target-excluded thresholds were 0.50 or 0.55. They produced zero EZ F1 and balanced accuracy 0.5, so the thresholded classifier collapsed to the negative class rather than merely suffering mild calibration error.

## Frozen matrix diagnostics

| Intervention | AUROC | AP | MRR | Delta AUROC vs Full | Delta AP vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full replay | 0.497126 | 0.339976 | 0.462141 | 0 | 0 |
| Shuffle patient reference within patient | 0.497211 | 0.340918 | 0.469882 | +0.000084 | +0.000942 |
| Permute temporal states and rebuild matrices | 0.516837 | 0.343531 | 0.486682 | +0.019711 | +0.003554 |

Shuffling the patient reference does not reduce performance. The reference branch therefore supplies no measurable stable localization information. Destroying temporal order improves AUROC and MRR, which is direct evidence that the learned cross-state geometry is harmful rather than useful on this cohort.

The data also expose a fixed design mismatch: all 256 records end before +30 seconds, so the sixth LATE state is invalid everywhere. The model is nominally 6x6 but one complete row/column is always masked. This was handled exactly as specified, without interpolation, but it weakens the proposed representation.

## Bootstrap

The 10,000-draw seed42 patient-cluster bootstrap gives SCM AUROC 95% CI `[0.437580, 0.556750]` and AP CI `[0.277475, 0.407660]`. A formal paired SCM-A1 bootstrap is not estimable because only public A1 fold/overall aggregates remain; matched A1 patient-query cells are absent. `ICTAL_BOOTSTRAP.csv` therefore reports SCM cluster intervals and differences from a fixed public A1 point, explicitly marked **not paired**. Reconstructing a paired CI from aggregate A1 values would be fabrication.

## Failure attribution

The failure is not a marginal gate miss. The model is near random overall, collapses under thresholding, and is unstable across folds. The patient-reference component is demonstrably non-informative, while the true temporal ordering is actively detrimental. This supports failure of the **patient-reference and cross-state matrix interpretation**. The evidence does not isolate the raw spectrum, self-comparison, or cross-seizure pooling individually because the protocol prohibited retrained ablations before a passing Full gate.

The architecture has 6,353 trainable parameters, used no GPU, and achieved about 18,975 channel scores/s in the real-patient CPU benchmark. Runtime was not the problem; the representation was.

**Terminal: `STOP_SCM_ICTAL_GATE_FAILED`.**

Per protocol, no ablation training, architecture revision, outer test, or Omni experiment was started.
