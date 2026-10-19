# Six-feature F0, no head interaction: fourteen approved configurations

**Completed and independently verified.** All14groups x5folds and Outer are complete:55fresh fits/670complete epochs,15exact reused cells,1120patient metric rows and10080independent endpoint checks. Primary patient-average main Macro-F1=0.6538278218104221, AUROC=0.7522551973793955, AP=0.5852396946163818. See the [Chinese findings and all nine metrics](FINDINGS.md), [complete English report](results/REPORT.md), [full-precision table](results/MAIN_RESULTS.csv), [paired intervals](results/INTERVALS.csv) and [verification](results/INDEPENDENT_OUTER_VERIFICATION.json). No independent-confirmation claim is made.

User-approved matrix excludes previous list items3 (isolated Late Gate deletion) and16 (interaction add-back). All fourteen remaining configurations complete the original five patient-held-out folds and Outer, irrespective of internal validation performance.

The reference permanently removes Variance and hidden Raw x physiology interaction. It is exactly the previous six-feature `WO_HEAD_INTERACTION`, not a newly selected model. The six measurements are beta, low gamma, partial high gamma80–125Hz, RMS, line length and spectral entropy10–125Hz; two summaries each. Original7/14-wide inputs are retained with Variance masked in every explicit pathway. All other groups also have no head interaction.

| Group | Relative change | Fresh fits |
| --- | --- | ---: |
| FULL_F0 | Exact previous six-feature WO_HEAD_INTERACTION | 0: five exact reuses |
| WO_EARLY_FILM | Remove early conditioning only | 5 |
| WO_BOTH_GUIDANCE | Remove early FiLM and late Gate, retain direct summaries | 5 |
| WO_BOTH_PR | Remove BOTH physiological-summary and final Raw96 patient Z | 5 |
| WO_RAW_PRE_REFERENCE | Remove Raw pre-ictal latent subtraction | 5 |
| EARLY_FILM_MISMATCH | Only early donor trajectory mismatched, within patient/seizure/identical support | 5 |
| RAW_ONLY | No explicit physiology; Raw PR and pre-reference retained | 0: five exact reuses |
| PHYSIOLOGY_ONLY | Twelve active summary coordinates, Feature PR retained | 0: five exact reuses |
| DROP_BETA | Mask beta in summary, early FiLM and late Gate | 5 |
| DROP_LOW_GAMMA | Mask low gamma in all three pathways | 5 |
| DROP_PARTIAL_HIGH_GAMMA | Mask80–125Hz input in all three pathways | 5 |
| DROP_RMS | Mask RMS in all three pathways | 5 |
| DROP_LINE_LENGTH | Mask line length in all three pathways | 5 |
| DROP_SPECTRAL_ENTROPY | Mask10–125Hz entropy in all three pathways | 5 |

Total70selected model/fold cells:55fresh fits plus15exact reuses. Reuse requires matched initial state and bit-exact original legal-validation replay for all fifteen reference cells. Other prior ablations still contained interaction and are NOT relabeled as these new variants. Previously completed seven classifiers x two Feature-PR settings are unchanged; their anonymous results are supplementary historical context only, not new fits or members of the fourteen-group CI family.

## Execution

Trusted server code and private runtime are both on D, reading existing caches in place. `run_all.ps1` owns a lock and sequential workers. Keep the foreground SSH owner alive on this Windows server; do not detach and disconnect. A failed worker stops the coordinator with preserved checkpoints/logs; diagnose before resuming with a new RunTag. Complete-epoch model/optimizer/RNG checkpoints and frozen per-patient inference caches support engineering-only exact resume.

```powershell
# Admission performs no formal model fits and no new Outer inference.
powershell -NoProfile -File D:\m8_6d_no_interaction_ablation_seed42_v1\run_all.ps1 -StopAfter Admission -RunTag admission01
# After source push and remote-tip verification:
powershell -NoProfile -File D:\m8_6d_no_interaction_ablation_seed42_v1\run_all.ps1 -SourceCommit VERIFIED_40_HEX_COMMIT -RunTag production01
```

Original80patients/7635channels/256seizures/1471965valid Raw250windows, seed42 and five roles remain unchanged. FP32, TF32off, original optimizer/loss/budget, full-patient updates and4096-window chunks are unchanged. One64-hidden head with matched initialization/slices/scaling; no trained-reference initialization. Legal internal validation alone selects epoch and threshold. All70selections freeze before new Outer, all70score cells seal before metrics. No Development admission gate, hyperparameter search, data extraction, Raw copy, subsampling or post-Outer changes.

## Complete outputs

Patient-mean Macro-F1/AUROC/AP/EZ-F1/NEZ-F1/ACC/MRR/Top-1/EZ-fraction bias; separate pooled metrics, five folds and sample SD, source summaries with small-group suppression, and complete coverage. Seed42paired10000-patient-bootstrap:14absolute +13reference-minus-ablation contrasts across9metrics=243nominal95% intervals;13additional approximate Bonferroni-level Macro-F1 intervals. Independent verification of1120patient rows and10080values. Because isolated Late Gate deletion was excluded, no full early-by-late factorial interaction is identifiable or reported.

The reference choice followed historical Outer viewing. Every result is **exploratory repeated-test**, not independent confirmation or proof of global feature optimality. Only source, hashes and anonymous aggregates/reports may be released. Raw data, identities, predictions, weights, preprocessors, maps and logs remain on the trusted server.

Actual admission on2026-10-19: PASS, five CPU mathematical tests, fifteen bit-exact selected-reference validation replays and three complete-largest-FIT-patient optimizer steps for each of eleven changed configurations. These33admission steps were discarded before formal fits; no completed formal epoch was discarded. Main11425parameters, Raw-only8433, Physiology-only1025. All55formal fits then completed without engineering restart. See [admission](results/ADMISSION.json), [execution audit](RUN_AUDIT.md), [actual completion status](results/OUTER_STATUS.json) and [figure QA](FIGURE_QA.md).

`verify_public.py`, `render_figures.py`, `write_findings.py` and `verify_git_release.py` operate only on anonymous public artifacts. They were added outside the frozen training/evaluation source snapshot, without changing any admitted scientific source byte. Reproduce local aggregate validation/figures/findings with these scripts after restoring the released results directory. The separate figure manifest reflects actual visible QA; rerendering initially marks fresh images pending review.
