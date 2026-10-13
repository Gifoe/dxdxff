# Frozen mechanism verification

All interventions use the selected **B3 backbone**, frozen alpha/beta, nuisance fit and original fold thresholds. They do not retrain or remove original input views. M1 is not independently trained B0; M2 is not independently trained B2.

| Frozen B3 inference | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| Full COVE | 0.716295 | 0.512349 | 0.642981 | 0.439535 | 0.689704 |
| M1 correction coefficients zero | 0.716164 | 0.512244 | 0.642810 | 0.439297 | 0.689704 |
| M2 incorrect correspondence | 0.716339 | 0.511922 | 0.642810 | 0.439297 | 0.689704 |
| M3 DELTA correction disabled | 0.716218 | 0.512333 | 0.642810 | 0.439297 | 0.689631 |
| M4 RATIO correction disabled | 0.716312 | 0.512301 | 0.642981 | 0.439535 | 0.689789 |

Full−M1 AUROC is+0.000130423,95% patient-ID cluster interval[-0.000082108,0.000360944]. Full−M2 AUROC is−0.000044425, interval[-0.000544752,0.000326414]. Neither supports useful module dependence or correct-view advantage. Removing RATIO corrections slightly **raises** AUROC/MRR. No threshold is reoptimized.

Disabling the module changes296 of352,239 within-patient channel pairs (pooled0.08403%; equal-appearance mean0.07679%), changes no Top1 decisions, and leaves MRR exactly unchanged. Wrong pairing changes340 pairs (pooled0.09653%) but also no Top1 decisions. Full versus M1 crosses the selected threshold for only one additional true EZ channel across the65 appearances; there is no FP change. The B3-versus-B0 specificity loss therefore is not fixed by disabling the module.

All14 individual group-correction-disable diagnostics preserve original ABS/DELTA/RATIO inputs. Beta/low/high gamma groups have no measurable AUROC/AP/Macro-F1 effect at the reported precision. RMS/variance effects are roughly1e-5–2e-5 AUROC, entropy6e-5–7e-5, and line-length corrections slightly harm AUROC/AP. These are descriptive small perturbations, not retrained feature-removal evidence or clinical causal effects. `GROUP_INTERACTION_EFFECTS.csv` reports every group, not a selected subset.

Validation mean absolute DELTA/RATIO corrections averaged across fold/groups are0.000633/0.000496 in post-preprocessed input units; maximum group-fold averages0.009236/0.004858. Alpha/beta, their bounded effective coefficients, FIT-OOF/validation correction magnitudes, correct/wrong q variances and near-zero rates are in `COVE_COEFFICIENTS.csv`. Nonzero parameters are not proof of useful physiology.

Most of the B3 AP point gain over B0 persists with M1 disabled; all MRR/Top1 point gain persists. This is consistent with backbone optimization/checkpoint-selection differences, not demonstrated incremental multiplication. The independently trained B2 comparison also fails to favor B3 AUROC. Nonzero conditional residual variation exists; useful exploitation of correctly paired residual products has not been established.
