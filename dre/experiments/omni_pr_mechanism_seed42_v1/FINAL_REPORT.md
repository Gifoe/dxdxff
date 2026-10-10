# Omni-PR mechanism identification, seed42

Terminal: `MECHANISM_INCONCLUSIVE`. PR gate: False; alignment gate: True.

| Arm | Patient AUROC | AP | MRR | Top1 | NDCG | Macro-F1 FIT tau | Pathological F1 FIT tau | Macro-F1 at0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.762163 | 0.477255 | 0.607458 | 0.508065 | 0.670700 | 0.610352 | 0.313503 | 0.545188 |
| C1-PR | 0.811823 | 0.523527 | 0.658966 | 0.564516 | 0.708337 | 0.655819 | 0.392346 | 0.644327 |
| C1-ABS | 0.813627 | 0.523435 | 0.662949 | 0.556452 | 0.708477 | 0.626587 | 0.349214 | 0.612537 |
| C1-SHUFFLE | 0.753240 | 0.467298 | 0.610012 | 0.508065 | 0.666687 | 0.603600 | 0.299685 | 0.544134 |

All summaries are patient-equal across the same124 patients/10202 labeled canonical channels. C0/PR are exact historical replay, not retrained candidates or new confirmations. Each arm uses its frozen fold-specific FIT-selected threshold;0.5 results are separate. Published pooled Task2 AUROC0.8061 is not used.

## Registered paired patient contrasts

| Contrast | AUROC delta |95% percentile CI | Positive folds | Positive patients |
|---|---:|---|---:|---:|
| C1-PR-C1-ABS | -0.001804 | [-0.012650, +0.008456] | 2/5 | 57/124 |
| C1-PR-C1-SHUFFLE | +0.058583 | [+0.027768, +0.089873] | 4/5 | 70/124 |
| C1-ABS-C0 | +0.051464 | [+0.020823, +0.082478] | 4/5 | 67/124 |
| C1-SHUFFLE-C0 | -0.008923 | [-0.018694, +0.001094] | 1/5 | 46/124 |
| C1-PR-C0 | +0.049660 | [+0.019098, +0.080574] | 4/5 | 68/124 |

10,000 unique-patient paired draws,seed42,all numeric thresholds and models fixed. Patient effects median/IQR and per-center contrasts are provided. Intervals are exploratory and do not correct historical cohort reuse or broader research-selection bias. No additional hypothesis test or outcome-based model search was run.

## Source decomposition

| Source | n | PR AUROC | ABS AUROC | SHUFFLE AUROC | PR-ABS | PR-SHUFFLE |
|---|---:|---:|---:|---:|---:|---:|
| HUP | 10 | 0.828366 | 0.813922 | 0.787480 | +0.014444 | +0.040886 |
| Open-iEEG | 103 | 0.820289 | 0.822967 | 0.756205 | -0.002679 | +0.064084 |
| SourceSink | 11 | 0.717520 | 0.725900 | 0.694350 | -0.008380 | +0.023170 |

## Frozen inference diagnostics

For a fixed beta, PR=(beta/sigma_patient) dot h + patient-constant offset. Removing mean subtraction at inference cannot alter exact within-patient ranking. Direct float64 formulas and pairwise ordering were verified using the historical float32 patient moments. Sigmoid-saturation ties are resolved using logits for this mathematical diagnostic, not by changing primary historical scores.
Scale-only maximum mathematical rank-metric error=0.0; maximum difference against original operational PR logit-ranking metrics=0.0; difference against archived probability-ranking metrics=0.0.
Scale-only Macro-F1=0.451420, delta historical PR=-0.204400, at the original PR threshold with no refit.
Frozen aligned PR head with inference-shuffled inputs: AUROC=0.667529, AP=0.280432, Macro-F1=0.541716. This differs from C1-SHUFFLE,which was trained from scratch on shuffled embeddings.

## Answers to26 requested questions

1. Yes. Historical C0/PR logits,selected checkpoint tensors,thresholds and patient metrics replay exactly; HISTORICAL_REPRODUCTION.json gives measured errors.
2. Yes. All124 patients,137 records and10202 labeled patient-channels retained with original five folds and labels.
3. ABS-C0 AUROC=+0.051464; AP=+0.046181. See registered intervals before claiming an improvement.
4. PR-ABS AUROC=-0.001804,CI[-0.012650,+0.008456]; full PR increment gate=False.
5. PR-SHUFFLE AUROC=+0.058583,CI[+0.027768,+0.089873]; alignment gate=True.
6. PR-ABS positivefolds=2/5; PR-SHUFFLE=4/5. Full fold metrics retained.
7. PR-ABS AP=+0.000092; PR-SHUFFLE AP=+0.056230. AUROC and AP are not interchangeable.
8. FIT-threshold Macro-F1 PR/ABS/SHUFFLE=0.655819/0.626587/0.603600; no test reoptimization.
9. FIT-threshold pathological F1 PR/ABS/SHUFFLE=0.392346/0.349214/0.299685.
10. PR-ABS MRR/Top1 deltas=-0.003983/+0.008065; PR-SHUFFLE=+0.048954/+0.056452.
11. The complete predictive incremental PR gate does not pass; do not claim patient normalization is necessary.
12. The registered trained-alignment control supports a contribution from correct embedding correspondence.
13. Source bias is constant within patient and cannot affect exact patient ranking; classification can change. All three heads have identical source-bias topology and loss regularization.
14. Scale-only mathematical AUROC/AP invariance passed; error=0.0. Operational/saturation differences are measured above,not silently replaced.
15. PR versus ABS compares patient-specific normalization to patient-equal globally standardized embeddings. It does not individually identify variance scaling,training centering,optimization conditioning or their interaction with learned beta and frozen logits.
16. See every source PR-minus-ABS/SHUFFLE effect above and complete center metrics; no center or patient was dropped and no source-specific threshold fitted.
17. Yes. HUP10 and SourceSink11 are small samples; center effects are descriptive and cannot establish broad source generalization.
18. ABS-C0 AUROC gain=+0.051464,compared to historical PR gain=+0.049660; descriptive comparability rule=False.
19. Train/evaluate SHUFFLE-C0 AUROC=-0.008923,AP=-0.009957. A surviving point estimate alone is not proof of nonspecific optimization; use its uncertainty and the matched alignment contrast.
20. Yes. Inner ABS scalers use inner-training patients only. Final scalers use outer-FIT only. All good channels enter moments equally within each equally weighted patient; between-patient variance is retained.
21. Yes.35 trainable parameters each; global moments are fitted statistics,not learned parameters. Original PR was never retrained.
22. The paired bootstrap does not establish the full registered PR hypothesis under all prespecified requirements.
23. Yes for outer patients. The original fold CNN and new head both exclude them. Inner head-validation patients were previously used by the frozen CNN; inner head OOF is not end-to-end OOF.
24. Training centering,patient-specific variance scaling,conditioning and learned-beta/base-logit interactions remain causally unidentified. Inference scale-only invariance only rules out mean subtraction alone as a direct ranking effect for fixed beta.
25. No unqualified Patient-Relative mechanism claim is justified. Follow the terminal and uncertainty,not the largest point estimate.
26. Use the simplest matched aligned linear head justified by the registered comparisons; if uncertainty remains,retain both descriptions without selecting a new ensemble or rerunning.

## Registered gates

{
  "terminal": "MECHANISM_INCONCLUSIVE",
  "PR_mechanism_gate": {
    "AUROC_delta_ge010": false,
    "AUROC_CI_lower_positive": false,
    "AP_nondecline": true,
    "positive_folds_ge4": false,
    "MacroF1_loss_le005": true,
    "integrity": true
  },
  "embedding_alignment_gate": {
    "AUROC_delta_ge010": true,
    "AUROC_CI_lower_positive": true,
    "AP_strict_gain": true,
    "logits_labels_sources_unpermuted": true
  },
  "PR_gate_pass": false,
  "alignment_gate_pass": true,
  "ABS_better_with_CI": false,
  "descriptive_comparability_rule_pass": false,
  "exploratory_historical_reuse": true,
  "protocol_sha256": "37453f811edc2f70d873dbe87b04641a9956521c75e166b6652aee20779e6f48"
}

## Scope and provenance

Historical C0/PR are reused observations,not a new confirmation. No new CNN/PR training,raw signal extraction,feature regeneration,architecture/seed/threshold search or post-outcome rescue. Only30 inner control heads and10 selected-length refits were trained. All models and transformations froze before the two new outer arms were evaluated.
Primary scores preserve historical sigmoid conventions. Scale-only logit-ranking diagnostics are explicitly separate. All patient/clinical/electrode identities,labels,embeddings,predictions,checkpoints,scalers,permutations and optimizer state remain private. Public files contain aggregate outputs and hashes only.
Protocol SHA256 `37453f811edc2f70d873dbe87b04641a9956521c75e166b6652aee20779e6f48`. New pre-outer freeze `3a2503f435db73b53ea9faa039a90892d6a182604b3531e744dce929ff997250`. Historical published tree is pinned in PROTOCOL_LOCK.json.
