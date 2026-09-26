# CalibRank-MLP seed-42 exploratory result

**Outcome: no meaningful constructive gain.** The retained B0 reproduces the intended patient-relative MLP, but neither adapter comes close to historical CDEL. This is one seed on historical outer outcomes that had already been inspected; it is not fresh confirmatory evidence or a multiseed conclusion.

| Variant | Patient Macro-F1 | Δ vs B0 (pp) | Patient EZ-F1 | Patient BA | EZ-AUPRC | EZ-MRR | Positive outer folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0, frozen original MLP | 0.61617 | — | 0.42387 | 0.66418 | 0.53200 | 0.73303 | — |
| B1, calibration | 0.61698 | +0.081 | 0.42208 | 0.66413 | 0.53200 | 0.73303 | 3/5 |
| B2, bounded ranking | 0.61704 | +0.087 | 0.42420 | 0.66458 | 0.53149 | 0.73302 | 2/5 |
| B3, both | 0.60581 | −1.036 | 0.39295 | 0.65525 | 0.53198 | 0.73204 | 1/5 |

Fixed-seed patient-paired 95% bootstrap intervals for Macro-F1 Δ vs B0, in percentage points: B1 `[-0.112, +0.288]`; B2 `[-0.104, +0.289]`; B3 `[-2.046, -0.158]`. B1/B2 intervals span zero and their point gains are under 0.1 pp; B3 is worse. B3 fold 4 is an especially stark validation-to-outer failure: validation Macro-F1 rises from `0.63990` to `0.64234`, but outer Macro-F1 falls from `0.59564` to `0.53947`. No outer result was used to change the fixed method.

## Answers to the preregistered questions

- **A — B0 reproduction:** Yes. The historical seed-42 B0 Macro-F1 is `0.616167`; its original five checkpoints were replayed with maximum channel-score discrepancy about `1.5e-7`. See `BASELINE_AUDIT.md`.
- **B — diagnostic headroom:** A label-using per-patient oracle shift gives `0.70260` Macro-F1 (+8.64 pp); label-using true-K with B0 ranking gives `0.64925` (+3.31 pp). Neither is deployable. Observed historical BCR ranking has EZ-AUPRC `0.56231` versus B0 `0.53200`, a +3.03 pp difference; it is not an identical-input causal ablation. See `ORACLE_DIAGNOSTICS.md`.
- **C — B1:** Macro-F1 rises only 0.081 pp, while EZ-F1 declines 0.179 pp and BA is effectively unchanged. As required by a constant within-patient shift, EZ-AUPRC and MRR are exactly unchanged. The oracle's large calibration headroom is not recovered by this label-blind head.
- **D — B2:** Macro-F1 rises only 0.087 pp, with 2/5 positive folds. EZ-AUPRC falls 0.051 pp and MRR is unchanged to five decimals. The intended ranking benefit is not observed.
- **E — B3:** No. Macro-F1 drops 1.036 pp and EZ-F1 drops 3.091 pp, without ranking improvement. Its validation-selected calibration does not transfer reliably, especially fold 4.
- **F — comparison with historical CDEL:** The best new variant, B2, remains about 2.755 pp below historical seed-42 CDEL Macro-F1 `0.644589`. The 80 patients, five outer folds, channel keys and label/evaluation semantics align; CDEL's constituent feature sets and training differ, so this is a historical reference, not a strictly identical-input architecture comparison.
- **G — uncertainty concentration:** In a post-hoc tertile analysis, 63/74 B1 and 41/52 B2 changed channel decisions fall in the highest B0-uncertainty tertile, as designed. That concentration does **not** yield a useful patient-equal gain: B1's high-uncertainty-patient tertile improves only about +0.25 pp, and B2's corresponding tertile is slightly negative. B3 changes many more decisions and harms patient-equal Macro-F1. The aggregate-only stratification is `results/UNCERTAINTY_DIAGNOSTIC.csv`.

`results/DEVELOPMENT_RESULTS.csv` records canonical validation selections; `results/OUTER_FOLD_RESULTS.csv`, `results/OUTER_SUMMARY.csv`, and `results/PAIRED_BOOTSTRAP.csv` contain outer aggregates. The original 20 cells are complete. A post-training `pandas.query` scope error affected only aggregate generation; `FINALIZATION_REPAIR.md` documents the non-scientific repair. All private patient/channel predictions, embeddings, caches, checkpoints and logs remain on the server.

**Interpretation:** `CALIBRANK_SINGLE_MODEL_GAIN_NOT_SUPPORTED_SEED42`. The oracle result diagnoses possible headroom but is not evidence that this small learned adapter can realize it. Do not claim superiority over B0 or CDEL from this run.
