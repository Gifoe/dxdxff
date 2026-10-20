# Actual result: directional PR ablations, not guaranteed stable superiority

All numbers below are actual original 80-patient / 7,635-channel / 256-seizure patient-held-out fivefold Outer results, seed 42. They are not 33-patient internal-development results. All 45 selected model/preprocessor/threshold cells were frozen before new Outer; all 45 prediction cells were sealed before metrics. This is exploratory repeated-test on the historically viewed cohort.

## The requested numerical direction is observed

| Configuration | Patient Macro-F1 | Patient AUROC | AP | EZ-F1 |
| --- | ---: | ---: | ---: | ---: |
| Original exact 0.653828 main | 0.653828 | 0.752255 | 0.585240 | 0.467424 |
| Fixed new Full | 0.646339 | 0.765888 | 0.589527 | 0.447095 |
| w/o Feature PR, both summary and window context | 0.636037 | 0.744249 | 0.575278 | 0.448413 |
| w/o final Raw 96D PR | 0.630820 | 0.763988 | 0.563932 | 0.431853 |
| w/o BOTH PR | 0.516322 | 0.759725 | 0.570728 | 0.371850 |

Full versus the original reference: Macro-F1 **-0.749 percentage points**, AUROC **+1.363 percentage points**. Thus Full remains above 0.64 and within the accepted 1-percentage-point Macro-F1 loss. AP improves 0.429 points, but EZ-F1 falls 2.033 points; this is not an across-endpoint improvement. MRR falls from 0.845674 to 0.836034 and Top-1 from 0.800 to 0.775. All nine endpoints, including NEZ-F1, ACC and EZ-fraction bias, are in [MAIN_RESULTS.csv](results/MAIN_RESULTS.csv).

The three PR removals all lower AUROC and Macro-F1 point estimates. NoPR was not deliberately damaged: meaningful original FIT-global physiology, original raw evidence, matched initialization, capacity, loss and training/selection rules remain. Five genuinely fresh joint-NoPR fits exactly reproduce the original joint-NoPR selected weights, validation logits, epochs and thresholds. See [independent selection replay](results/INTERNAL_VERIFICATION.json) and [fresh-control identity](results/NULL_CONTROL_RETRAIN_PARITY.csv).

## The stable-contribution objective is not established

Full minus the specified control; 10,000 seed-42 patient-paired bootstrap draws, nominal 95% intervals:

| PR removal | Macro-F1 difference [95%interval] | AUROC difference [95%interval] | Folds both directions positive |
| --- | --- | --- | ---: |
| Feature PR | +0.010301 [-0.008479,0.028906] | +0.021639 [0.004156,0.040276] | 3/5 |
| Raw PR | +0.015519 [-0.002714,0.034562] | +0.001900 [-0.023610,0.027679] | 3/5 |
| BOTH PR | +0.130017 [0.090533,0.171814] | +0.006163 [-0.016667,0.029613] | 3/5 |

The joint-PR Macro-F1 effect is clear in this conditional nominal interval, but its AUROC interval crosses zero; folds 3 and 4 have lower Full AUROC than joint-NoPR. The Raw-PR AUROC effect is only 0.190 points and is uncertain. Feature-PR AUROC has a positive nominal interval, but the additional 14-comparison interval crosses zero. No primary component passes the preregistered simultaneous two-endpoint/four-of-five-fold rule. Therefore “deleting PR reliably lowers both metrics” is not a supported general claim, even though this run's overall point estimates have that direction.

See [all 162 nominal intervals](results/INTERVALS.csv), [14 additional-family intervals](results/PRIMARY_FAMILY_INTERVALS.csv), [component audit](results/COMPONENT_EVIDENCE.csv) and [fixed objective audit](results/GOAL_AUDIT.json). Its performance conditions pass, but its stable retained-component condition fails. There is only one training seed and no independent confirmation cohort.

## Unsupported modules must not be presented as effective

| Fixed control | Macro-F1 | AUROC | What it permits us to say |
| --- | ---: | ---: | --- |
| w/o relative window PR | 0.652906 | 0.764621 | New window context adds only 0.127 AUC points while losing 0.657 Macro-F1 points; no clear added value. |
| w/o early FiLM | 0.646647 | 0.770064 | Removing early FiLM has slightly better point estimates on both endpoints; an effectiveness claim is unsupported. |
| w/o late Gate | 0.653465 | 0.765286 | Removing late Gate improves Macro-F1; the tiny opposite AUROC difference does not establish two-metric utility. |
| w/o own pre-reference | 0.618218 | 0.743705 | Both point estimates decline; Macro-F1 uncertainty is more favorable than AUROC uncertainty. |

Full versus original B0 jointly changes summary-PR numerical handling and window guidance. The window-only control specifically prevents attributing the overall AUROC increase to the new window mechanism. The nominal pre-reference AUROC interval also narrowly crosses zero. Early/late removals change parameter count andcannot isolate information content from capacity/optimization.

Cross-source stability is not improved uniformly: Full versus original has LZU Macro-F1 0.567027versus0.594778andpediatric AUROC 0.748240versus0.766708. These are descriptive subgroup results,not fresh confirmatory tests. Across-fold sample SD grows from0.032932to0.044402forMacro-F1and0.039583to0.052450forAUROC. All patients remain in the primary averages.

No favorable ablation is renamed as the main model after viewing Outer. In particular,the promising no-window/no-late point estimates are exploratory observations,not a new independent selection. Any simplification/optimization would require a separately frozen,legal training/internal-validation selection procedure; these Outer results cannot be used to rescue a threshold orclaim global feature/module optimality.

## Completion and release evidence

- 40 actual fresh fits / 505 completed epochs; 5 exact historical reference cells reused.
- 45 selected models replayed/frozen; all 45 Outer score cells sealed before metrics.
- 720 private patient rows / 6,480 endpoint values independently recomputed on server; 162 nominal / 14 additional intervals verified, maximum discrepancy 8.881784197001252e-16.
- Pre-Outer freeze SHA256 `ad4579246ea7d636641ce387ded8b42dadb71b952ac5845c5c9160da1301073e`; score-seal SHA256 `bd5f968c46bdfce99c6a1c585ef8f95290832d7b4648de594d9264793bc05746`.
- Actual 61,010-byte anonymous 30-member export SHA256 `7641a9552dee43b041ca31422ea51906ceb1360d7b260d458a1506c3967a1a31`; strict independent public-package checks pass and all 29 payload hashes match after extraction.
- Only source, hashes, anonymous aggregate reports and audits are public. Raw signals, identities, individual predictions, preprocessors, weights, caches and runtime logs remain on the trusted server.
