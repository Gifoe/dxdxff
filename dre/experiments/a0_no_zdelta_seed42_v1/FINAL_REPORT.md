# A0 w/o ZDELTA — seed42

Completed 2026-10-13. Terminal: **NO_EVIDENCE_OF_IMPROVEMENT_ON_MATCHED_OUTER**.

Removing ZDELTA did not improve the matched outer localization result: AUROC decreased by 0.002791 and AP by 0.007235. Neither change is statistically resolved by the fixed patient-cluster bootstrap. Do not interpret the intervals crossing zero as proof of equivalence or proof that ZDELTA is unnecessary.

## Outer: 80 unique held-out patients

Primary unit is equal-patient mean, not pooled channels or equal-fold mean. Each patient appears in exactly one of the five outer folds; 7,635 unique patient-channels, 1,743 EZ and 5,892 NEZ.

| Metric | Original A0 | A0 w/o ZDELTA | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|
| AUROC | 0.739768 | 0.736977 | -0.002791 | [-0.013290, 0.006292] |
| AP / EZ-AUPRC | 0.537887 | 0.530653 | -0.007235 | [-0.020354, 0.006002] |
| Macro-F1 | 0.624919 | 0.624236 | -0.000684 | [-0.009662, 0.008299] |
| EZ-F1 | 0.419405 | 0.414671 | -0.004734 | [-0.019539, 0.009966] |
| NEZ-F1 | 0.830434 | 0.833800 | 0.003366 | [-0.001143, 0.007917] |
| Balanced accuracy | 0.666405 | 0.663990 | -0.002414 | [-0.013400, 0.009116] |
| MRR | 0.728620 | 0.694267 | -0.034353 | [-0.089027, 0.022209] |
| Top1 | 0.637500 | 0.587500 | -0.050000 | [-0.137500, 0.037500] |
| Sensitivity | 0.471033 | 0.460046 | -0.010987 | [-0.034504, 0.012348] |
| Specificity | 0.861776 | 0.867935 | 0.006158 | [-0.000622, 0.013349] |
| Accuracy | 0.758773 | 0.763172 | 0.004399 | [-0.001902, 0.011014] |
| Predicted EZ fraction | 0.207411 | 0.203485 | -0.003925 | [-0.011430, 0.003104] |

The pooled confusion matrix is a **secondary descriptive channel-level statistic**, not the primary patient-mean metric:

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Original A0 | 742 | 834 | 5058 | 1001 |
| A0 w/o ZDELTA | 729 | 799 | 5093 | 1014 |

Net change: 35 fewer false positives but 13 additional missed EZ channels. At patient level, specificity increased 0.006158 while sensitivity decreased 0.010987. MRR and Top1 also decreased; the confidence intervals do not establish a reliable difference.

## Development: 65 validation-fold appearances from 47 IDs

This is the exact historical development unit. Repeated IDs keep their original fold appearances; it must not be reported as a 65-independent-patient test. A separate unique-ID summary is supplied in `results/DEVELOPMENT_UNIQUE_ID_SUMMARY.csv`.

| Metric | Original A0 | A0 w/o ZDELTA | Difference | Patient-cluster 95% CI |
|---|---:|---:|---:|---:|
| AUROC | 0.710667 | 0.716569 | 0.005903 | [-0.015128, 0.023940] |
| AP / EZ-AUPRC | 0.518235 | 0.508559 | -0.009676 | [-0.026131, 0.003740] |
| Macro-F1 | 0.638080 | 0.642067 | 0.003987 | [-0.004433, 0.012576] |
| EZ-F1 | 0.431198 | 0.433516 | 0.002318 | [-0.011554, 0.016311] |
| NEZ-F1 | 0.844961 | 0.850618 | 0.005656 | [0.001024, 0.010953] |
| Balanced accuracy | 0.665436 | 0.669667 | 0.004231 | [-0.004692, 0.013424] |
| MRR | 0.686550 | 0.676591 | -0.009959 | [-0.064969, 0.046181] |
| Top1 | 0.584615 | 0.553846 | -0.030769 | [-0.117647, 0.053571] |
| Sensitivity | 0.465904 | 0.464334 | -0.001571 | [-0.018554, 0.014851] |
| Specificity | 0.864968 | 0.875001 | 0.010033 | [0.002184, 0.019239] |
| Accuracy | 0.773174 | 0.779216 | 0.006042 | [-0.000414, 0.013144] |
| Predicted EZ fraction | 0.198578 | 0.191949 | -0.006630 | [-0.014621, 0.000789] |

The small development AUROC increase (+0.005903) did not transfer to outer. Development AP decreased. No architecture, selection rule or threshold was changed in response to these results.

## Exact intervention and controls

- Original **source-free A0**, not source-conditioned D2: 8,817 trainable parameters in both arms. Original network `88 → LayerNorm88 → Linear96 → GELU → Dropout0.15 → Linear1`; no source tensors, no PURN, spline or alternative normalization.
- **Patient Z is retained.** ABS, DELTA and RATIO are retained. Exact original Patient Z, FIT-only imputer and scaler are reused bitwise; raw feature values and original five-fold identities are unchanged.
- Names are parsed by the hash-locked historical `ablation_core.py`, not inferred from numerical column position. All **26 ZDELTA columns** are zeroed after original preprocessing; the 62 retained columns are bitwise identical. All 88 input positions and native LayerNorm topology remain.
- This is the historical zero-mask ablation definition, not a 62D reduced-width network. Masking also changes the input-dependent LayerNorm statistics; the result is not an isolated causal estimate of physiological information alone.
- Five scratch training cells use identical original shared initialization and seed `42+1009*fold`; original optimizer, patient-update loss, patience and validation-only threshold selection. No extra seeds or hyperparameter searches.
- Original A0 checkpoints and scores are reused, not retrained or overwritten. Baseline legal validation logits/probabilities replay bitwise and original thresholds replay exactly. Both models independently retain their validation-selected fold thresholds.
- `MODEL_SELECTION.csv` and its frozen selection records use `bank_sha256` for the new experiment's private alignment/masked bank in both rows; it is **not** a claim that historical A0 was trained on that masked bank. Baseline original training-bank hashes are separately verified in `BASELINE_REPRODUCTION.json` (`binding.dependencies.baseline_bank*`), and original input-bank hashes appear in `PRETRAIN_AUDIT.json`. Baseline scoring uses the original unmasked inputs/sealed predictions.
- All five new checkpoints, masks, FIT preprocessors and thresholds were locked before new outer features. All five new outer score files were sealed before loading the global clinical labels. Existing baseline outer files were whole-file hash checked.
- CPU outer inference uses the original baseline outer backend. CPU/CUDA validation replay satisfies predeclared logit `2e-6` and probability `2e-7` bounds, with exact threshold decisions.
- Bootstrap: 10,000 paired draws, seed42; unique patient-ID clusters, repeated development appearances retained; no threshold reoptimization, no retraining inside draws. Unadjusted percentile 95% intervals; the many metrics are descriptive, not multiplicity-adjusted discoveries.

## Checks and provenance

`PRETRAIN_AUDIT.json`: PASS, 70 executed engineering assertions including label-free FIT-only perturbation, exact untouched initialization and source-free architecture, original input scaling, and synthetic deterministic CPU checkpoint/model/optimizer/RNG resume.

`INDEPENDENT_VERIFICATION.json`: PASS for all 290 patient-fold/model metric cells (130 development + 160 outer), all primary/fold/center/unique-ID tables and 32 bootstrap intervals. Independent sklearn versus original metric formula drift ≤3.34e-16; independent bootstrap multiplicity calculation matches to 1e-12. Verification reads sealed scores and does not run fresh inference.

`POSTHOC_VERIFICATION.json`: PASS; independent name-based 26-column mask, unchanged original banks/labels/splits/preprocessor statistics and all retained values, plus all secondary pooled classification statistics (drift ≤8.33e-17). This reporting-only script was added after scoring; scientific code, checkpoints and thresholds were not edited.

Runtime: PyTorch 2.11.0+cu128. Frozen feature archive, name order, manifest and original split hashes are in `PROTOCOL_LOCK.json` and the executed audits. The inherited feature export contract is 80 patients / 7,635 channels / 256 seizures; this run consumed its legal aggregated 88D banks, not new raw EEG or seizure extraction. The 256-seizure count is historical export provenance, not a new waveform-level enumeration.

The five new training cells completed 45 total epochs in 4.650 seconds as measured around the original trainer, selected epoch metadata and history collection. This excludes protocol construction, admission checks, bootstrap, independent verification and transfer time. No native crash or training restart occurred.

## Interpretation and stop

This evidence does **not** support removing ZDELTA to improve A0. The apparent development AUROC gain is small and uncertain; outer ranking and EZ retrieval move down, while fewer predicted EZ channels reduce FP at a sensitivity cost. It does not prove that ZDELTA is indispensable, because the paired confidence intervals include zero.

Both development and outer are exploratory repeated evaluations on previously used cohorts. They are not a blind confirmation or a newly untouched external test. Seed42 only; no v2 or outcome-tuned rerun. Individual patient/channel identities, predictions, feature banks, weights, raw signals and logs remain on the trusted server. Only source, hashes and anonymous aggregate results are published.
