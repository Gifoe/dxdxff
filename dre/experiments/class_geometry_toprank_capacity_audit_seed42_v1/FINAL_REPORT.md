# Class geometry, top-rank and capacity audit (seed 42)

Development-only FIT/validation experiment; no outer-test loader, prediction or performance was used. Historical outer exposure in earlier project work means these are exploratory development findings, not sealed confirmation.
A1 reproduction: 150/150 checkpoints, maximum grid error 0.0, VLOO Macro-F1 0.6259962097. Six balanced-loss unit tests and all-patient class-support checks passed.

| Matched effect | Delta EZ-AUPRC | Delta EZ-MRR | Delta Top1 | Delta Macro-F1 | AUPRC-positive folds | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| class_balanced_bce | -0.003466 | +0.002817 | -0.015385 | +0.002228 | 1/5 | FAIL |
| hard_negative_on_original | -0.015509 | -0.009304 | -0.015385 | +0.002502 | 1/5 | FAIL |
| hard_negative_on_class_balanced | -0.001277 | -0.006895 | -0.015385 | -0.005061 | 3/5 | FAIL |
| feature_capacity_64 | -0.029008 | -0.068266 | -0.123077 | -0.003565 | 1/5 | FAIL |
| feature_capacity_128 | -0.023892 | -0.049758 | -0.092308 | -0.005267 | 1/5 | FAIL |

## Direct answers

1. True patient-level class balancing changes EZ-AUPRC by -0.003466; CLASS_GEOMETRY_RANKING_BOTTLENECK_NOT_SUPPORTED.
2. Original-loss-mass imbalance versus AUPRC gain has Pearson -0.1048, Spearman -0.0661. Quartile aggregates are in `audit_a_class_geometry/EZ_FRACTION_QUARTILES.csv`; correlation is descriptive, not selection.
3. AUPRC delta is -0.003466 and Macro-F1 delta is +0.002228; class balance is not a ranking improvement under the gate.
4. Hard-negative MRR deltas are -0.009304 on original BCE and -0.006895 on balanced BCE; Top1 deltas are -0.015385 and -0.015385.
5. Hard-negative AUPRC deltas are -0.015509 and -0.001277; preservation checks are explicit in `HARD_NEGATIVE_GATE.json`.
6. Matched B11−B10 versus B01−B00 results do not justify claiming complementary benefit unless the prelocked matched gate passes; overall terminal: HARD_NEGATIVE_RANKING_NOT_SUPPORTED.
7. Hard-negative margin deltas are +0.058965 and +0.026499. This is a label-using diagnostic only.
8. Capacity 64/128 validation AUPRC deltas are -0.029008 and -0.023892; FEATURE_ENCODER_UNDERCAPACITY_NOT_SUPPORTED.
9. CAP1/CAP2 FIT AUPRC deltas are +0.014448 and -0.013649; corresponding train–validation gap changes are +0.043456 and +0.010243. FIT-only gains, if present, indicate generalization difficulty rather than simple under-capacity; this cannot prove a feature ceiling.
10. No mechanism has matched causal support sufficient to name a ranking source.
11. Source A1 Macro-F1 is 0.625996. Best candidate among the five matched rows is hard_negative_on_original at 0.628498, with AUPRC delta -0.015509. Candidates jointly reaching ≥0.640 with nondecreasing AUPRC and MRR: none.
12. No single new model redesign is justified from this audit alone.

Optional first-positive diagnostic: run; MRR delta -0.010279. It is not eligible as a final candidate.
Exact terminal: `RANKING_SOURCE_STILL_UNRESOLVED`.
OUTER_TEST_ACCESSED = NO
