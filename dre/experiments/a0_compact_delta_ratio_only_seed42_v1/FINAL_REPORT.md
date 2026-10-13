# Compact A0: DELTA Only and RATIO Only, seed42

## Outer results (primary: mean over the same 80 patients)

| Input | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| F0: ABS + DELTA + RATIO (historical) | 0.736977 | 0.530653 | 0.624236 | 0.414671 | 0.694267 | 0.587500 |
| ABS Only (historical F2) | 0.699011 | 0.482819 | 0.605307 | 0.390291 | 0.652969 | 0.537500 |
| **DELTA Only (new)** | **0.726333** | **0.517057** | **0.624497** | **0.422314** | **0.699468** | **0.575000** |
| **RATIO Only (new)** | **0.714943** | **0.512992** | **0.612709** | **0.393439** | **0.709198** | **0.637500** |

These are actual completed five-fold runs, not renamed old ablation numbers.
The previous "without DELTA" used ABS + RATIO; "without RATIO" used ABS + DELTA.
Those artifacts remain unchanged. No Raw-only model was rerun.
Here DELTA denotes the deterministic reference-difference feature view, **not the 1–4 Hz delta frequency band**.

| Input | NEZ-F1 | Balanced accuracy | Sensitivity | Specificity | Accuracy | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 0.833800 | 0.663990 | 0.460046 | 0.867935 | 0.763172 | 0.203485 |
| ABS Only | 0.820322 | 0.639921 | 0.433662 | 0.846180 | 0.739020 | 0.214719 |
| DELTA Only | 0.826680 | 0.669411 | 0.488251 | 0.850571 | 0.752569 | 0.216585 |
| RATIO Only | 0.831978 | 0.643720 | 0.411685 | 0.875754 | 0.755485 | 0.181364 |

Mean confusion counts per patient (not a pooled confusion matrix):

| Input | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| F0 | 9.1125 | 9.9875 | 63.6625 | 12.6750 |
| ABS Only | 8.6375 | 11.8125 | 61.8375 | 13.1500 |
| DELTA Only | 9.3625 | 11.0250 | 62.6250 | 12.4250 |
| RATIO Only | 7.8250 | 8.9875 | 64.6625 | 13.9625 |

## Development (validation, not outer TEST)

Primary development unit is equal fold appearance: 65 appearances of 47 distinct patients.

| Input | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| F0 | 0.716569 | 0.508559 | 0.642067 | 0.433516 | 0.676591 |
| ABS Only | 0.687139 | 0.456533 | 0.617246 | 0.397969 | 0.612188 |
| DELTA Only | 0.716600 | 0.504414 | 0.642347 | 0.443493 | 0.695850 |
| RATIO Only | 0.709422 | 0.512282 | 0.624367 | 0.405565 | 0.686817 |

The secondary equal-47-ID AUROCs, after averaging repeated appearances within ID, are F0 0.723153, ABS Only 0.690675, DELTA Only 0.736154, RATIO Only 0.722631. The separate file `DEVELOPMENT_UNIQUE_ID_SUMMARY.csv` contains this estimand; it must not be substituted for the primary development mean.

## Paired uncertainty and interpretation

Patient-ID paired cluster bootstrap: 10,000 draws, seed42, fixed predictions and fixed fold thresholds, unadjusted descriptive 95% percentile intervals.

| Outer comparison | AUROC difference | 95% interval |
|---|---:|---:|
| F0 minus DELTA Only | 0.010644 | [-0.004112, 0.023260] |
| F0 minus RATIO Only | 0.022034 | [-0.000676, 0.042983] |
| ABS Only minus DELTA Only | -0.027322 | [-0.065049, 0.008485] |
| ABS Only minus RATIO Only | -0.015932 | [-0.056521, 0.021338] |
| DELTA Only minus RATIO Only | 0.011390 | [-0.007420, 0.029733] |

DELTA Only has the best single-view AUROC point estimate. Neither single view exceeds the full F0 AUROC or AP. DELTA's Macro-F1 exceeds F0 by only 0.000261; the paired interval for that difference crosses zero. RATIO has higher MRR/Top1 point estimates but lower sensitivity; ranking-top performance does not imply better overall discrimination or localization recall. The listed AUROC intervals all include zero, so these data do not establish a statistically resolved ordering between the views. No additional configuration was selected or trained after reading outer outcomes.

## Exact matched implementation and executed audits

- 80 patients, 256 seizures, 7,635 canonical patient-channels, 1,743 EZ, 5,892 NEZ; all original five-fold memberships unchanged.
- New runs: DELTA Only retains 26 original coordinates (including original auxiliary duplicates), RATIO Only retains 18. Remaining coordinates are zeroed **after** original full-88D Patient Z and FIT-only imputer/scaler. An 88D interface, original LayerNorm, and the exact source-free 8,817-parameter A0 factory remain unchanged. Coordinate counts are not counts of independent informative features.
- Original trainer imported unchanged: CUDA AdamW, lr0.001, weight decay0.0001, original FIT class-weighted NEZ-positive BCE, one update per FIT patient, clipping1, maximum30 epochs, patience6. Each arm independently selects checkpoint and threshold by the original validation Macro-F1/EZ-F1 rules. Fold seed is 42 + 1009 × fold.
- Selected epochs: DELTA 2/3/2/1/3; RATIO 2/3/3/1/6. Selected NEZ thresholds: DELTA 0.420/0.290/0.340/0.380/0.400; RATIO 0.410/0.235/0.355/0.425/0.385. All ten selections were frozen before new outer inference.
- Training completed 86 epochs across ten cells. Sum of measured trainer invocations was 8.03 seconds on the server; provenance, transfers, inference, and independent verification are additional elapsed work. Per-cell measurements are in `RESOURCE_AUDIT.csv`.
- Pretraining audit passed manifest/name/mask checks, original FIT/VAL and scaler provenance, historical F0/ABS checkpoint and score-seal checks, source-code invariance, and exact synthetic CPU model/optimizer/history/RNG resume. No GPU bitwise-training claim is made.
- Validation CPU/CUDA replay passed predeclared logit2e-6 / probability2e-7 bounds with identical threshold decisions and exact selected validation metrics. Outer feature inference uses the historical CPU backend.
- All ten new outer score files were sealed before the evaluation process opened the full label array. Historical outer predictions were hash-verified and reused, not regenerated.
- Independent score-only sklearn verification passed all 580 patient/appearance metric cells, aggregate/fold/center tables, pooled-channel secondary statistics, and independently recomputed bootstrap intervals. Source and transferred aggregate hashes match the executed server artifacts.

Both the earlier and current outer cohort have been viewed. This is an **outer-informed exploratory repeated-cohort ablation**, not a new blind confirmation. Full preprocessing uses unlabeled within-patient channel context; scaler fitting and checkpoint/threshold selection use only FIT/VAL. No raw EEG, EDF, new feature extraction, other seeds, or test-selected thresholds were used.

Only code, hashes, protocol, anonymous aggregate CSVs and audits are public. Patient/channel identities, predictions, checkpoints and runtime logs remain on the trusted server. Seed42 stops here.
