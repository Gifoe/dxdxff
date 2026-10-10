# Omni-D2 seed42: frozen patient-level outer evaluation

Terminal: `SOURCE_SHUFFLE_CONTROL_NOT_BEATEN`. Registered primary continuation gate: False.

| Model | Patient AUROC | AP | MRR | Top1 | NDCG | MacroF1 FIT tau | PathF1 FIT tau | MacroF1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.762163 | 0.477255 | 0.607458 | 0.508065 | 0.670700 | 0.610352 | 0.313503 | 0.545188 |
| C1 | 0.811823 | 0.523527 | 0.658966 | 0.564516 | 0.708337 | 0.655819 | 0.392346 | 0.644327 |
| C2 | 0.807748 | 0.520349 | 0.659480 | 0.564516 | 0.706867 | 0.655314 | 0.390798 | 0.640308 |
| C3 | 0.806116 | 0.516928 | 0.655236 | 0.556452 | 0.704348 | 0.657716 | 0.393288 | 0.651207 |

All124 unique patients and10202 labeled patient-channels are included exactly once per arm. Original137 records/five folds retained. Ranking is within-patient, summaries patient-equal; published pooled0.8061 is not a criterion. Classification columns distinguish fixed0.5 and own FIT-selected numeric thresholds. All CLASS/RANK metrics and center/fold tables are in CSVs.

C0@0.5 historical AUROC0.762163/AP0.477255/MacroF10.545188; old A1Context AUROC0.763012/AP0.472740/MacroF10.509492 is historical only. No old Ictal numbers are numerically compared to Omni.

## Paired bootstrap

| Contrast | Patient AUROC delta |95% percentile CI | Positive folds | Positive patients |
|---|---:|---|---:|---:|
| C1-C0 | +0.049660 | [+0.019098, +0.080574] | 4/5 | 68/124 |
| C2-C0 | +0.045585 | [+0.015300, +0.077009] | 4/5 | 61/124 |
| C2-C1 | -0.004075 | [-0.011148, +0.003201] | 2/5 | 40/124 |
| C2-C3 | +0.001632 | [-0.005742, +0.009002] | 2/5 | 42/124 |

10,000 paired seed42 unique-patient draws. Fixed checkpoints and numeric thresholds, no bootstrap reoptimization. Nominal intervals are descriptive and do not adjust repeated historical cohort viewing or research selection. This is not an untouched external test.

## Source effects

| Source | n | C0 AUROC | C2 AUROC | C2-C0 | C2-C3 |
|---|---:|---:|---:|---:|---:|
| HUP | 10 | 0.805939 | 0.831128 | +0.025189 | +0.014043 |
| Open-iEEG | 103 | 0.764318 | 0.817131 | +0.052813 | +0.002793 |
| SourceSink | 11 | 0.702193 | 0.698637 | -0.003556 | -0.020524 |

## Answers to26 required questions

1. Original TimeConv reproduced: fresh bank hard gate passed every historical patient metric; scores within1e-6, no archived-score substitution. BASELINE_REPRODUCTION.json reports exact maximum error.
2. All124 patients,137 records and original folds preserved; label orientation pathological1/normal0 and unknown masks unchanged.
3. Only frozen CNN32D embeddings plus original frozen logits are used;35/105 parameter correction heads, no CNN optimization or raw feature redefinition.
4. C1 patient AUROC=0.811823, delta C0=+0.049660.
5. C2 patient AUROC=0.807748, delta C0=+0.045585.
6. C2 AP=0.520349, delta C0=+0.043094.
7. C2 matched FIT-tau MacroF1=0.655314, delta C0=+0.044962; fixed0.5=0.640308.
8. C2 pathologicalF1=0.390798, delta matched C0=+0.077295.
9. C2 MRR/Top1=0.659480/0.564516; deltas=+0.052023/+0.056452.
10. C2-C1 AUROC=-0.004075; registered requirement +0.005=False.
11. C2-C3 AUROC=+0.001632; matched-capacity source placebo beaten by +0.005=False.
12. All five folds,45 inner heads,15 selected-epoch full FIT refits and four OOF arms completed.
13. Positive C2-C0 AUROC folds=4/5.
14. Source effect table above reports the Open-iEEG majority and both smaller sources without exclusion; do not infer general source transfer from pooled or majority performance.
15. HUP10 and SourceSink11 AUROC changes are reported above. Small-source gate passed=True. Their uncertainty is substantial.
16. Mean patient C2 source-direction RMS=0.214668; meaningful-direction gate=False. Source_PARAMETER/SOURCE_DIRECTION_MATRIX and source correction CSVs give all singular values, ranks, norms, biases, variances and RMS ratios.
17. Mean direction-off pair reversal=0.047465; full C2-vs-C0 pair reversal=0.249420.
18. Removing source bias changes matched-threshold MacroF1 by -0.011657; within-patient ranking deltas are exactly0 by constant-translation audit. Bias can affect classification, not patient ranking.
19. Direction-off AUROC/AP/MacroF1 deltas=+0.001801/-0.002419/-0.006248. This is same-model inference diagnosis, not retrained ablation.
20. All added corrections off recovers C0 probabilities exactly; classification intervention uses C2 numeric threshold, unlike separately FIT-selected C0 threshold. No false claim of classification equality across different thresholds.
21. Bootstrap intervals above quantify patient sampling variation. Intervals including0 do not clearly distinguish gains. Even positive nominal intervals do not constitute independent validation or causal source evidence.
22. Yes: inner head-validation patients were previously used to train the corresponding frozen outer CNN. Inner head-OOF is NOT full-pipeline OOF. Only outer held patients are excluded by both CNN and head.
23. No outer labels enter head initialization, loss, inner epoch/threshold selection or source shuffling; predictions are atomically sealed before final label access.
24. Patient AUROC>=0.7722: True. This target is exploratory, not external validation.
25. The registered gate fails; this evidence does not justify joint fine-tuning as a rescue of this mechanism.
26. Stop this source-direction family under the current prespecified experiment; no post-outcome rescue.

## Gate and limitations

{
  "C2_C0_AUROC_ge010": true,
  "C2_C1_AUROC_ge005": false,
  "C2_C3_AUROC_ge005": false,
  "AP_nondecline": true,
  "matched_FIT_threshold_MacroF1_nondecline": true,
  "pathological_F1_decline_le010": true,
  "AUROC_positive_folds_ge4": true,
  "small_source_AUROC_decline_le050": true,
  "meaningful_direction": false,
  "MRR_Top1_nondecline": true,
  "cohort_checkpoint_alignment_integrity": true
}

Unknown good channels enter label-blind per-patient normalization, never loss/metrics. Head CV uses only FIT labels and one common three-way split. C3 has identical initial tensors/parameter count/budget but source-category assignment only is shuffled; imbalanced Open-iEEG composition makes many assignments unchanged. Changed-source fractions are disclosed, not forced to impossible100% derangement.
This transfers the operational source-correction mechanism, not Ictal weights or biological meaning. Head-only in-sample-CNN inner validation can be optimistic; thresholds may not transfer to outer held scores. Inference interventions do not identify effects of retraining or prove source causality. Unknown-source fallback retains shared beta only.
Bias-off ranking reuses full frozen ordering to avoid finite-precision/sigmoid saturation ties while verifying constant source offset. Classification is recomputed at the frozen C2 threshold. Effective U@V numerical rank uses float64 SVD; zero vectors have no assigned cosine.
Private waveforms, labels, patient/channel identities, embeddings, scores, head/CNN tensor checkpoints, optimizers and logs remain on server. Public output is aggregate-only. No raw EEG extraction, no CNN training, no architecture/seed sweep, no test-derived threshold.

## Provenance

Protocol SHA256 `25e9c5b18e44ef7176915b5b5d8252e03437236c43a4e422d342373b487e807b`; pre-outer head freeze `e3c8db5e67ecbe9f3b5dc36793eb28d6176acb5c3fdbe0aede7eaa3bfa1d720c`. All prediction and model hashes retained privately; compact integrity verification delivered separately.
