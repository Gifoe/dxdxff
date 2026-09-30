# PRCD-EZ seed-42 final report

This is validation-only historical fixed-query VLOO development. No outer-test record was opened. The locked Ictal gate failed, so Omni was not started.

| Model | Selected config | Params | AUROC | AP | Macro-F1 | MRR | Top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical A1 | fixed reference | historical | 0.746382 | 0.576743 | 0.620810 | 0.740038 | 0.654771 |
| CD-EZ | K128 H1 | 4,161 | 0.680595 | 0.484880 | 0.545221 | 0.663231 | 0.557797 |
| PRCD-EZ | K128 H1 | 5,121 | 0.730063 | 0.550435 | 0.610955 | 0.721537 | 0.627618 |

PRCD-EZ improved substantially over CD-EZ (AUROC +0.049468), so patient-relative motif coordinates are useful inside this design. That does not make the model competitive: against A1, PRCD-EZ lost 0.016319 AUROC, 0.026308 AP, 0.018501 MRR and 0.027153 Top1. Only 2/5 folds exceeded the corresponding historical A1 fold AUROC.

## Locked gate

- AUROC >= 0.751382: **FAIL** (0.730063)
- AP >= 0.576743: **FAIL** (0.550435)
- positive AUROC folds >= 3/5: **FAIL** (2/5)
- MRR >= 0.730038: **FAIL** (0.721537)
- Macro-F1 >= 0.605810: **PASS** (0.610955)

## Mechanism diagnostics

All diagnostics reuse the frozen selected epoch and threshold. Thresholds were not re-optimized. The lightweight H1 was deterministically replayed only because per-epoch weights had not been persisted; the replay matched every frozen prediction within 1.79e-7 before interventions.

| Intervention | AUROC | AP | Macro-F1 | Delta AUROC | Delta AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full replay | 0.730063 | 0.550435 | 0.610955 | 0 | 0 |
| Shuffle dictionary across patient channels | 0.479655 | 0.274963 | 0.477444 | -0.250408 | -0.275472 |
| Remove D/R coordinates | 0.673218 | 0.480445 | 0.443384 | -0.056844 | -0.069990 |
| Remove all 30 biomarkers | 0.727100 | 0.549997 | 0.611030 | -0.002963 | -0.000438 |

The classifier uses channel-aligned dictionary features and relative coordinates. The bundled biomarkers add almost nothing. Power/PAC/AEC are not individually selected by mRMR in this protocol: all 30 are appended after dictionary selection, so claiming that one biomarker family was "selected most" would be false.

## Selected dictionary structure

Across five folds (640 selected B2 dictionary coordinates):

- coordinates: rank R 509, absolute A 112, median deviation D 19;
- temporal views: original 413, first difference 227;
- record pools: mean 200, Q75 151, max 149, median 140;
- most frequent statistics: margin 89, winner-k2 77, positive-winner-k2 74, winner-k3 65, positive-winner-k3 57;
- kernel-specific selections by dilation: dilation 4 = 211, dilation 2 = 193, dilation 1 = 146; another 90 margin/run features summarize mixed-dilation groups;
- most frequent groups: 17 (64), 5 (48), 37 (44), 19 (42), 6 (35).

The original view contributes more than the difference view. Rank coordinates dominate; simple median deviation is rarely selected.

## Runtime

- real 115-channel/3-record speed gate: 5.23 s total (3.15 s base features, 2.08 s dictionary);
- 80-patient shared base cache: about 30 s active parallel compute with 8 workers;
- parallel fold dictionary extraction: 159-172 s observed wall per clean fold;
- mRMR: 0.55-0.63 s per fold;
- all predeclared H0/H1 candidate training: 78-89 s per fold, with five folds run concurrently;
- CPU only; peak GPU memory 0;
- observed peak process RAM about 1.31 GB and peak five-process aggregate about 6.4 GB.

## Scientific conclusion

This is Scenario D. Fixed motifs alone are much worse than A1. Relative coordinates recover part of that loss but do not restore A1 ranking or precision. Biomarkers do not close the gap. The fixed-dictionary route should stop; automatically producing a v2 would be outcome-driven iteration.

**Terminal: `STOP_ICTAL_GATE_FAILED`.**
