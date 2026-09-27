# Zero-shot Patient Geometry Recovery Study — development-only

65 fixed VLOO cells, 47 unique patients, 20 fixed 50/50 query repetitions. B=0 inference; no candidate pool used.

1. Exact A1 reproduction: yes, 150/150 checkpoint/R4 replay within 1e-6; fixed-query AP 0.576743.
2. Z1 direction canonicalization AP 0.567973, delta -0.008770.
3. Z2 class-conditional alignment AP 0.577667, delta +0.000924. Z2B cross-patient centroid-SupCon AP 0.574908.
4. Z3 patient-heldout first-order MLDG AP 0.556248, delta -0.020495.
5. Z4 FIT-selected geometry plus MLDG AP 0.558115, delta -0.018629.
6. Best descriptive zero-shot variant `Z2_CLASS_CONDITIONAL_ALIGNMENT` AP 0.577667, paired delta +0.000924 [-0.000917,+0.003408], positive folds 3/5.
7. Any variant reaches original B8 0.599632 under predeclared gate: False; descriptive best-AP row gate: False.
8. Any variant reaches best B8 0.605291 under predeclared gate: False; descriptive best-AP row gate: False.
9. Patient-specific-minus-shared headroom at common epoch 30: A1 0.257646; best 0.256190. Smaller supports, but does not prove, canonicalization.
10. Mean within-fold cross-patient direction cosine at common epoch 30: A1 0.289302; best 0.288942.
11. FIT-direction reversal rate at common epoch 30: A1 0.107692; best 0.107692.
12. Interpretation: `R4_LEVEL_ZEROSHOT_CANONICALIZATION_NOT_SUPPORTED`. Any richer-physiology switch is a next-study hypothesis, not a causal conclusion from these retrospective data.

## Primary matched-query performance

| Variant | EZ-AP | AUROC | MRR | Top1 | Macro-F1 | EZ-F1 | BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z0_A1 | 0.576743 | 0.746382 | 0.740038 | 0.654771 | 0.620810 | 0.407246 | 0.671972 |
| Z1_DIRECTION_CANONICALIZATION | 0.567973 | 0.747230 | 0.723801 | 0.632273 | 0.612165 | 0.393342 | 0.668187 |
| Z2_CLASS_CONDITIONAL_ALIGNMENT | 0.577667 | 0.745840 | 0.743431 | 0.660202 | 0.616315 | 0.397706 | 0.668570 |
| Z2B_CROSSPATIENT_SUPCON | 0.574908 | 0.750523 | 0.733500 | 0.644686 | 0.632500 | 0.424921 | 0.680562 |
| Z3_PATIENT_HELDOUT_MLDG | 0.556248 | 0.741401 | 0.712717 | 0.621412 | 0.616437 | 0.399090 | 0.667446 |
| Z4_GEOMETRY_PLUS_MLDG | 0.558115 | 0.741693 | 0.724418 | 0.637704 | 0.614188 | 0.390988 | 0.663759 |
| C1_PATIENT_ADV | 0.571286 | 0.743084 | 0.746010 | 0.668735 | 0.616870 | 0.401256 | 0.668281 |
| B8 best Teacher (reference only) | 0.605291 | — | — | — | — | — | — |

No B8 training, target adaptation, Student distillation or outer evaluation. The legacy loader materializes all 80 labels. Exact A1 VLOO creates cross-patient label dependencies, so strict target-label sequencing is false even though all candidate score/R4 grids were frozen before the new label-using pass. Geometry diagnostics use one comparable epoch-30 coordinate system per fold but do not inspect every per-cell selected checkpoint; a null epoch-30 diagnostic cannot exclude a mechanism at a different selected epoch. Descriptive best-variant selection is uncorrected for multiplicity.
