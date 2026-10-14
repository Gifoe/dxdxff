# DGM-Net A2 — Seed42 Development

| Model | Parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 1537 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 |
| A2 | 11019 | 0.729412 | 0.511697 | 0.650581 | 0.450204 | 0.659223 |
| A1 | 11024 | not run | not run | not run | not run | not run |
| A3 | 11019 | not run | not run | not run | not run | not run |
| A4 | 11019 | not run | not run | not run | not run | not run |

Terminal: `RAW_MORPHOLOGY_MINIMUM_INCREMENT_NOT_ESTABLISHED`.

Only A2 was requested in the final scope amendment: five trained cells; A0 exact frozen replay. A1/A3/A4 were not trained. Thus the full window-guidance research hypothesis cannot be accepted, regardless of A2 performance. No new Outer evaluation, rescue variants or other seeds.

Primary estimand: 65 original validation appearances/47 unique IDs; patient-equal, NEZ-positive scoring. Validation labels selected models and thresholds, as in the historical protocol. Repeated Development is exploratory; percentile intervals condition on those selections and do not include their uncertainty. Unique-ID secondary summaries do not replace the primary estimand.

## A2 versus frozen D1

- ez_auroc: 0.729412; difference +0.003435.
- ez_auprc: 0.511697; difference +0.008096.
- macro_f1: 0.650581; difference +0.006035.
- ez_f1: 0.450204; difference +0.006667.
- sensitivity: 0.510055; difference +0.003810.
- specificity: 0.873587; difference +0.007514.
- ez_mrr: 0.659223; difference +0.000721.
- top1_is_ez: 0.523077; difference +0.000000.

## Paired bootstrap

| Metric | A2−A0 | Exploratory95% interval |
|---|---:|---|
| ez_auroc | +0.003435 | [-0.000732, +0.007621] |
| ez_auprc | +0.008096 | [+0.000524, +0.016335] |
| macro_f1 | +0.006035 | [+0.000814, +0.011477] |
| ez_f1 | +0.006667 | [-0.002232, +0.016452] |
| nez_f1 | +0.005403 | [+0.002085, +0.009656] |
| balanced_accuracy | +0.005662 | [-0.001194, +0.012961] |
| sensitivity | +0.003810 | [-0.010228, +0.019369] |
| specificity | +0.007514 | [+0.000121, +0.017135] |
| ez_mrr | +0.000721 | [-0.022939, +0.024027] |
| top1_is_ez | +0.000000 | [-0.046154, +0.044118] |
| accuracy | +0.006022 | [+0.002500, +0.009522] |
| predicted_ez_fraction | -0.002456 | [-0.008742, +0.004097] |

## Identity and mechanism limits

D1 checkpoints/preprocessing stayed frozen. Step0 final logits exactly equal D1; gate initialized exactly1. Legal-FIT gradient smoke model was discarded and training restarted from the locked initialization. No GRU/LSTM/attention exists. Reported gate interventions keep the trained CNN/head and A2-selected threshold; residual-disabled logits exactly recover D1, but its F1 need not equal A0 F1 because A0 uses its own threshold. All seven neutral-feature interventions are diagnostic only; no subset selected.

Multiplicative guidance forbids a direct additive physiology-to-output branch but cannot alone prove morphological complementarity. Wrong-window/static inference changes establish dependence, not predictive value versus retrained controls. Mean pooling is invariant to a joint within-stage permutation of Raw and physiology, so this model cannot establish within-stage chronological order. Bandwidth-limited high gamma stops at125Hz; metadata time0 is not independently verified clinical onset.

## Frozen selections

| Fold | Epoch | NEZ threshold | Alpha | Effective residual coefficient |
|---|---:|---:|---:|---:|
| 1 | 22 | 0.430 | 0.352127 | 0.169130 |
| 2 | 1 | 0.260 | 0.001488 | 0.000744 |
| 3 | 1 | 0.300 | 0.005876 | 0.002938 |
| 4 | 29 | 0.365 | 0.472128 | 0.219959 |
| 5 | 17 | 0.375 | 0.254089 | 0.124379 |

## Historical external architecture reference

Historical D2 (13,643 parameters, GRU) Development: AUROC0.732515/AP0.515027/Macro-F10.650238/EZ-F10.451028/MRR0.684968. Copied from the frozen historical report, not a newly trained comparison. D2 was not reused as A2 and no Outer individual outputs were opened.

See every fold/center, interventions/confusions, scales, bootstrap and independent verification in results. Only code/hashes/anonymous aggregate outputs are public; patient IDs, labels, Raw, checkpoints and per-channel scores remain on the original trusted server.

## Actual execution and conditioning-scale limitation

All five A2 training cells completed, 95 epochs total. Recorded epoch training/selection time sums to approximately115 seconds, excluding preparation/imports/tests/interventions/bootstrap. Local and trusted-server deterministic tests passed4/4; the separate actual-FIT two-update gradient/identity admission also passed. Every completed BEST/LAST checkpoint and all exported result hashes were rechecked. A preclinical missing source-root import was repaired before any new window feature or A2 training was produced; the original freeze/source and failed logs are retained privately and the public amendment records unchanged protocol/science.

The numerical conditioning floor was **chosen in this implementation** and locked before extraction/training, not specified numerically in the user attachment. Its universal absolute value1e-6 is too large relative to several measurements in native units: FIT typical absolute DELTA is roughly0.85–0.98e-10 for beta,1.53–2.51e-11 for lowgamma,1.07–1.97e-12 for partial highgamma, and2.14–2.53e-9 for variance. The floor binds these four families in all five folds. Their typical normalized magnitudes are consequently approximately1e-4,1e-5,1e-6 and.002, whereas RMS/line-length/entropy use the fitted typical magnitudes. This materially weakens a balanced seven-family conditioning interpretation. Neutralizing a suppressed coordinate without a metric change is **not evidence that the underlying physiological family is useless**.

No floor, threshold, feature subset or model was changed after observing outcomes. The reported numbers are the genuine frozen run, not a corrected-scale rerun. A future different-floor study would require a separately authorized, prospectively locked protocol; this report cannot support rejecting the general hypothesis of properly scaled physiology-guided morphology.

## Frozen intervention observations

| A2 intervention | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| Full |0.729412|0.511697|0.650581|0.450204|
| Gate identity |0.728468|0.510016|0.650043|0.449168|
| Wrong-window DELTA |0.728969|0.511576|0.650038|0.449178|
| Static stage DELTA |0.729180|0.511951|0.650190|0.449500|
| Residual disabled, A2 threshold |0.725978|0.503601|0.640465|0.436260|

Full−wrong-window Macro-F1 is only+.000543, and static AP is slightly higher than full. Thus frozen dependence is small and does not establish meaningful precise-window guidance. In folds2/3 residual coefficients are only.000744/.002938 and classifications remain identical to A0. Macro-F1 increases strictly in only3/5 folds. These findings do not pass the registered minimum; A2's AUROC and EZ-F1 paired intervals also include zero. AP/Macro-F1 intervals exclude zero descriptively but remain selected-Development, unadjusted intervals and do not override the failed effect-size gate or absent trained controls.
