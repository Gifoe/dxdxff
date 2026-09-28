# RawTiny patient-relative seed42 — final Stage B report

## Decision

**Terminal: `RAW_WAVEFORM_FRONTEND_NOT_SUPPORTED_UNDER_CURRENT_PROTOCOL`.** RawTiny-PR recovered a large part of the weak raw-only model, but no new B=0 model exceeded the exact A1 baseline or reached AP 0.590. This is an exploratory development study: historical target outcomes had already been viewed, and no outer/external test was run.

| Model | EZ-AP | ΔAP versus A1 | 95% patient-ID CI | Positive folds versus A1 |
| --- | ---: | ---: | ---: | ---: |
| M0 exact A1 | **0.576743** | — | — | — |
| M1 RawTiny, no PR | 0.336249 | −0.240495 | [−0.316662, −0.166846] | 0/5 |
| M2 RawTiny + PR | 0.429936 | −0.146807 | [−0.206801, −0.082947] | 0/5 |
| M3 Hybrid + PR | 0.557161 | −0.019582 | [−0.043621, +0.005018] | 2/5 |

The planned mechanism contrasts are real but do not rescue the final model: M2−M1 was **+0.093688 AP** (CI **[+0.042573, +0.152536]**, 5/5 folds); M3−M2 was **+0.127225 AP** (CI **[+0.065825, +0.187149]**, 5/5 folds). The first supports patient-relative processing *within these two raw models*; the second supports retaining engineered 36D information relative to RawTiny-PR. Neither contrast proves that raw waveform contains information missing from A1.

All comparisons used the same 65 VLOO development cells, 47 unique patient IDs, and 20 fixed-query repetitions; 10,000 bootstrap draws clustered by unique patient ID. There were 1,289 estimable AP queries of 1,300 per model. The exact A1 AP `0.5767434626151353` replayed.

## Answers in the requested order

1. **Exact A1 reproduced?** Yes. The historical source checkpoint grid and R4 replay had zero error, and the fixed-query AP replay was exactly `0.5767434626151353`.
2. **Raw coverage?** 80/80 A1 patients and 256/256 matched seizure/run records had aligned raw waveform. The audit found 24,995 run-channel incidences, 7,635 unique cohort channels and 1,471,965 matched windows; no raw-missing A1 patient was removed. Thus all four models used the common A1 cohort.
3. **Seizure/onset timing verified?** Channel/run/window-center matching and valid interval bounds passed. The source constructor places seizure onset at the 60 s raw midpoint, but the cache lacks an explicit centered flag and independent EDF-to-cache verification: **0/256 matched runs have independently verified onset provenance**. Do not claim stronger temporal validation.
4. **Parameters?** M1 **5,729**, M2 **32,993**, M3 **34,018** trainable parameters (A1 pathway frozen in M3). All are far below the suggested 100K–300K; this is a small-capacity test, not an implementation of that suggested range.
5. **M1 better than A1?** No: −0.240495 AP, CI wholly negative, 0/5 folds.
6. **M2 better than M1?** Yes: +0.093688 AP, CI wholly positive, 5/5 folds; MRR and Top1 also increased by +0.1691 and +0.1823.
7. **Does latent patient-relative z contribute?** Inference-only zero-z reduced M2 AP from **0.429936** to **0.296837** (−0.133100). For M3 it reduced AP from **0.557161** to **0.556001** (−0.001160). This is a model-input intervention, not a retrained component ablation, and may produce out-of-distribution representations.
8. **Does within-patient rank contribute?** Zero-rank changed M2 AP by **−0.000675** and M3 by **−0.000335**. Its measured effect is negligible under this inference-only intervention.
9. **M3 better than both?** It substantially exceeded M2 (+0.127225 AP, 5/5 folds) but **not A1** (−0.019582; CI crosses zero; only 2/5 folds positive).
10. **Are engineered features complementary to raw?** Relative to raw-only M2, the hybrid has a stable advantage. This supports the importance of the 36D/A1 path in this architecture; it does not demonstrate a useful *incremental raw benefit over A1*.
11. **Best zero-shot AP?** Exact original M0 A1, **0.576743**. Best *new* model M3, **0.557161**.
12. **Any AP ≥0.590?** No.
13. **Any at current B8 0.599632?** No.
14. **Any at best B8 0.605291?** No.
15. **Five-fold consistency?** M2>M1 and M3>M2 in 5/5, but M1 and M2 remained below A1 in 5/5; M3>A1 in only 2/5. Thus the requested final improvement over A1 is not stable.
16. **Center consistency?** M2−M1 and M3−M2 are positive descriptively in HUP, multi-site and LZU; M3−A1 is negative in all three aggregates (approximately −0.0031, −0.0386, −0.0218). No Fudan patient appears among these 47 unique target IDs, so four-center target robustness cannot be claimed.
17. **Shared versus patient-specific linear gap?** A FIT-only, fold-1 descriptive probe found M2 cross-seizure shared AP **0.4310** versus own-patient AP **0.6107** (gap **0.1797**); post-PR shared AP **0.5776** versus own-patient AP **0.5993** (gap **0.0217**). Thus the gap shrank substantially *within M2*. M1 cross-seizure gap was **0.1485**. The checkpoints had already been trained on all 51 fold-FIT patients, and own-patient probes use FIT labels; these are not independent zero-shot estimates or evidence of target success.
18. **Likely bottleneck?** The data do **not** support the specific claim that replacing engineered input with this small raw waveform frontend resolves the localization bottleneck. M1 was far worse; M2's patient-relative gain only partially repaired it; M3 stayed below A1. Limited model capacity and unresolved raw onset provenance prevent a universal conclusion that raw physiology lacks useful information. A missing spatial/propagation signal or inadequate raw representation remains possible, but is not established here.

## Additional mechanism checks and limitations

In M3, mean |α| was **0.02860** and the raw-residual/engineered-embedding RMS ratio was only **0.02133**. At inference, setting α=0 reduced AP from **0.557161** to **0.553578** (−0.003583): the raw residual is used, but its measured contribution is small and the hybrid still underperforms A1. The α=0 model retains its separately trained PR correction, so it is **not** an exact replay of A1. The *initial* zero-α, zero-output-residual checkpoint did replay A1 exactly (maximum logit difference 0).

The strict FIT-only epoch-selection requirement conflicted with the inherited exact A1 VLOO rule. The pre-outcome lock explicitly selected matched VLOO: other validation patients' labels can affect a patient's epoch/threshold choice, never its own label. Hence this is not independent sealed confirmation. The legacy dataset constructor materializes validation labels before score freezing, although the new snapshot code did not index them; the 450 score files were hash-frozen before their target metrics were computed. Posthoc interventions used target labels only for diagnostics after the primary outcome, without retraining or model choice. No outer test, new model, larger search, or multiseed run followed these results.
