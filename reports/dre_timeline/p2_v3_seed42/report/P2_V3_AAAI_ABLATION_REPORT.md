# P2/V3 AAAI Ablation Report

The 0.90/0.10 probability fusion was locked after observing the current seed-42 development OOF results. All alternative weights and operators are ablation-only and cannot select a new formal model.

Formal predictions use one fold-global threshold selected only on that fold's validation patients. True-K is diagnostic and is never used for formal prediction.

## 1. Component Evolution

P0 availability: `available`; no-Q10 availability: `UNAVAILABLE_MISSING_INPUT`.

## 2. Branch Contribution

P2-Q10, V3-QBC, and the locked fusion are evaluated on exactly the same aligned channels.

## 3. Fusion Weight Sensitivity

The fixed grid is descriptive only; lambda=0.10 remains the locked configuration regardless of observed scores.

## 4. Fusion Operator

Probability, logit, and within-patient rank fusion use the same fixed 0.90/0.10 weights.

## 5. Score-vs-Threshold Decomposition

T1 changes only the score; T2 additionally adapts the validation threshold.

## 6. Shuffled-V3 Negative Control

V3 scores are permuted only within patient and separately for validation and test.

## 7. Fold and Center Stability

See the by-fold and by-center CSV files and paired figures.

## 8. Paired Bootstrap

Bootstrap resampling is patient-level with replacement.

## 9. Complementarity Analysis

Correlation, disagreement, changed-channel, and near-boundary files are descriptive only.

## 10. Reproduction and Leakage Audit

```json
{
  "status": "passed",
  "checks": {
    "p2_legal": true,
    "fusion_legal": true,
    "p2_truek": true,
    "fusion_truek": true,
    "fold_1_threshold": true,
    "fold_1_f1": true,
    "fold_2_threshold": true,
    "fold_2_f1": true,
    "fold_3_threshold": true,
    "fold_3_f1": true,
    "fold_4_threshold": true,
    "fold_4_f1": true,
    "fold_5_threshold": true,
    "fold_5_f1": true
  },
  "folds": [
    {
      "outer_fold": 1,
      "expected_threshold": 0.45,
      "actual_threshold": 0.45,
      "expected_macro_f1": 0.5977687039294951,
      "actual_macro_f1": 0.5977687039294951,
      "passed": true
    },
    {
      "outer_fold": 2,
      "expected_threshold": 0.365,
      "actual_threshold": 0.365,
      "expected_macro_f1": 0.666682456264136,
      "actual_macro_f1": 0.666682456264136,
      "passed": true
    },
    {
      "outer_fold": 3,
      "expected_threshold": 0.425,
      "actual_threshold": 0.425,
      "expected_macro_f1": 0.651833731109791,
      "actual_macro_f1": 0.651833731109791,
      "passed": true
    },
    {
      "outer_fold": 4,
      "expected_threshold": 0.43,
      "actual_threshold": 0.43,
      "expected_macro_f1": 0.6200849848758997,
      "actual_macro_f1": 0.6200849848758997,
      "passed": true
    },
    {
      "outer_fold": 5,
      "expected_threshold": 0.445,
      "actual_threshold": 0.445,
      "expected_macro_f1": 0.6779716539230917,
      "actual_macro_f1": 0.6779716539230918,
      "passed": true
    }
  ],
  "actual_p2_legal": 0.6284640770231468,
  "actual_fusion_legal": 0.6432651653484063,
  "actual_p2_truek": 0.6487672375803516,
  "actual_fusion_truek": 0.6503983337462481,
  "legal_tolerance": 0.0005,
  "truek_tolerance": 0.0005
}
```

## Key Results

| experiment | analysis_status | patient_macro_f1 | patient_ez_f1 | patient_nez_f1 | patient_balanced_accuracy | patient_ez_auprc | patient_ez_auroc | patient_ez_mrr | top1_is_ez_rate | predicted_ez_count_mae | predicted_ez_fraction_mae | truek_patient_macro_f1 | n_patients | n_channels | true_ez_fraction | predicted_ez_fraction | valid_ez_auprc_patients | valid_ez_auroc_patients | absolute_delta_vs_p2 | relative_delta_vs_p2 | ci_low | ci_high | probability_delta_gt_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1_P2_Q10_ONLY | ABLATION_ONLY_NOT_FOR_MODEL_SELECTION | 0.6284640770231468 | 0.4303441475289965 | 0.826584006517297 | 0.6652685203287666 | 0.5211757734041642 | 0.740946917213695 | 0.6886285891121532 | 0.5625 | 12.1875 | 0.12623748653219832 | 0.6487672375803516 | 80.0 | 7635.0 | 0.22829076620825148 | 0.22082514734774067 | 80.0 | 80.0 | 0.0 | 0.0 | nan | nan | nan |
| B2_V3_QBC_ONLY | ABLATION_ONLY_NOT_FOR_MODEL_SELECTION | 0.6225383624058317 | 0.42867992218908446 | 0.816396802622579 | 0.6710497027327393 | 0.5552540713183796 | 0.7328889767325487 | 0.775641728613139 | 0.6875 | 12.35 | 0.13066906492567468 | 0.6572080494563401 | 80.0 | 7635.0 | 0.22829076620825148 | 0.2361493123772102 | 80.0 | 80.0 | -0.005925714617315059 | -0.009428883581355137 | nan | nan | nan |
| B3_P2_Q10_V3_LOCKED_FUSION | EXPLORATORY_LOCKED_FUSION | 0.6432651653484063 | 0.45652550572091266 | 0.8300048249759001 | 0.6835376920076838 | 0.5340242781131186 | 0.7445908122028875 | 0.7464776809403334 | 0.6625 | 11.6625 | 0.12036392768818405 | 0.6503983337462481 | 80.0 | 7635.0 | 0.22829076620825148 | 0.23077930582842174 | 80.0 | 80.0 | 0.014801088325259504 | 0.023551208201697055 | 0.0069885219977265595 | 0.02285333022746724 | 1.0 |
