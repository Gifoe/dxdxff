# A0-SSS-MIL seed42 — completed exploratory development experiment

| Model | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0.638080 | 0.431198 | 0.844961 | 0.665436 | 0.518235 | 0.710667 | 0.686550 | 0.584615 | 0.465904 | 0.864968 | 0.773174 |
| D2 | 0.647926 | 0.443199 | 0.852653 | 0.669549 | 0.519936 | 0.718221 | 0.719610 | 0.615385 | 0.463999 | 0.875099 | 0.790571 |
| S0 | 0.611674 | 0.393463 | 0.829884 | 0.646417 | 0.479618 | 0.684511 | 0.680221 | 0.569231 | 0.455779 | 0.837056 | 0.754447 |
| S1 | 0.650392 | 0.446891 | 0.853893 | 0.672471 | 0.515275 | 0.719545 | 0.723804 | 0.615385 | 0.470185 | 0.874757 | 0.793284 |
| S2 | 0.650461 | 0.446891 | 0.854032 | 0.672607 | 0.514985 | 0.719614 | 0.723816 | 0.615385 | 0.470185 | 0.875029 | 0.793520 |
| S1_RAW_DISABLED | 0.650355 | 0.446568 | 0.854142 | 0.672586 | 0.515047 | 0.719666 | 0.723816 | 0.615385 | 0.469719 | 0.875453 | 0.793692 |

Terminal: `CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN`. Continuation gate: False; desired 0.700 target: False.

Means weight 65 patient-fold appearances equally; 47 unique patient IDs, 6,273 channel appearances. Bootstrap resamples unique IDs and keeps all repeated appearances together. No outer TEST evaluation.

## Paired contrasts

| Contrast | Macro-F1 delta | ID-cluster 95% CI | Positive folds |
|---|---:|---|---:|
| S0-D0 | -0.026406 | [-0.054330, +0.002245] | 1/5 |
| S1-D0 | +0.012312 | [-0.003699, +0.030498] | 4/5 |
| S1-D2 | +0.002466 | [-0.002435, +0.007317] | 4/5 |
| S2-D2 | +0.002535 | [-0.002368, +0.007373] | 4/5 |
| S1-S2 | -0.000070 | [-0.000176, +0.000000] | 0/5 |

## Required answers

1. D0/D2 exact frozen metric replay passed after independent checkpoint/hash gates; no control retraining.
2. All 80 original patients, 7,635 canonical channels and five original split identities retained. Clinical labels unchanged.
3. 256 raw records, 24,995 run-channel incidences and 1,471,965 valid candidate windows. Three padded records have candidates inside measured intervals; no padding enters MIL.
4. Onset provenance remains independently verified for 0/256. This limits temporal interpretation to cache-relative information, not clinical propagation or onset identification.
5. This is a code-grounded SSS-inspired adaptation, not exact SSS reproduction; fixed31 measured patches, auxiliary projection and two-level masked MIL are documented.
6. S0 standalone Macro-F1 0.611674, EZ-AP 0.479618, AUROC 0.684511; compare the matched D0 row, not historical RawTiny.
7. S1-D2 Macro-F1 delta +0.002466; paired interval above.
8. S1-S2 Macro-F1 delta -0.000070; paired interval above.
9. S1-S2 is the channel-assignment control. Small or nonpositive differences do not distinguish true correspondence from capacity/fine-tuning; a positive difference is still not a causal physiological proof.
10. S1 EZ-AP 0.515275 versus D2 0.519936.
11. S1 EZ-F1 0.446891 versus D2 0.443199.
12. S1 sensitivity/specificity 0.470185/0.874757 versus D2 0.463999/0.875099.
13. Over 6,273 repeated channel appearances, S1 versus D2 gains 20 TP and loses 5 TP; adds 27 FP and removes 31 FP. It recovers 51 errors and spoils 32 correct decisions. With raw disabled, it recovers 52 errors and spoils 29. Turning raw on versus off adds 1 TP but also 5 FP; the improvement over D2 cannot be attributed to raw evidence.
14. S1 versus D2 pairwise ranking-reversal fractions by fold are 0.006095, 0.065004, 0.008176, 0.034041 and 0.006334. Mean absolute rank shifts are 0.5954, 4.6296, 0.6469, 2.7691 and 0.5497. Raw-disabled rankings change similarly: these shifts primarily reflect engineered-branch fine-tuning rather than correctly assigned raw evidence.
15. Material branch collapse under the prelocked numerical definition: True. The selected fold5 raw/engineered RMS ratio is 0.00003029, below the 0.0001 floor. The encoder is not constant: all five raw embedding variances exceed 0.10. Selected S1 gamma values are -0.000592, 0.010270, 0.000346, -0.000660 and -0.000031. Corresponding RMS ratios are 0.000899, 0.018253, 0.000424, 0.000740 and 0.000030. Disabling raw changes patient-equal Macro-F1 only from 0.650392 to 0.650355; only fold2 has any thresholded decision changes (0.4064% of its channel appearances).
16. S1 window attention has mean entropy 2.480766, effective window count 11.9511/12 and mean maximum weight 0.094036. No weights exceed 0.99, invalid weights are zero and no all-padding NaNs occur. Attention is nearly uniform, not concentrated or collapsed, and is not a validated biomarker.
17. S1 beats D2 in 4/5 folds, S2 in 0/5. All five folds were completed without outcome-based discarding.
18. S1-D2 Macro-F1 differences are HUP +0.004136, LZU +0.002263, Multicenter +0.004859 and Pediatric -0.003389. S1-S2 is zero for the first three groups and -0.000377 for Pediatric. There is no consistent advantage of correct raw assignment; no center-specific model or threshold was used.
19. +0.020 S1-D2 requirement: False; complete continuation gate: False.
20. S1 Macro-F1 >=0.700: False.
21. The full predeclared continuation gate does not pass; these results are exploratory repeated development, not independent confirmation.
22. Stop this seed42 protocol; do not add rescue components or retune on these outcomes.

## Limitations and control interpretation

S1's +0.002466 gain over D2 is 0.2466 percentage points, not the required +0.020 absolute gain. Its 95% interval crosses zero. S2 slightly exceeds S1, and S1 EZ-AP declines by 0.004661 relative to D2. The small gain does not support added correctly assigned raw information: the joint fine-tuning/capacity explanation is sufficient for this registered seed42 experiment. This result does not establish that raw physiology is generally uninformative.

S1/S2 start from D2 checkpoints previously selected on these VAL patients and then select again on VAL. This additional repeated-validation selection bias is disclosed; no independent held-out confirmation is claimed.
S1_RAW_DISABLED is an inference intervention at the S1-selected threshold using jointly fine-tuned engineered weights. It is not frozen D2 replay.
Shuffle is within identical valid-seizure signatures to preserve run membership and missingness. Singleton signatures remain unchanged and their counts are audited.
All-clinical-label feature-export member was never loaded; FIT/VAL labels came from frozen development banks. Raw trusted pickle contains clinical metadata fields that were ignored. Unlabeled raw preprocessing includes original cohort context, but no outer labels, predictions, or metrics are used.
Private raw tensors, identities, per-channel scores, checkpoints, optimizer states and execution logs remain on the original server.

## Execution verification

Protocol SHA256: `052c2c9e6fac8daad3e65449c6ebaf0fc989b26ef24bce5bca77814d9d3a31b7`. Fifteen cell completion hashes, thresholds and selected epochs are in MODEL_SELECTION.csv.
Total completed epochs: 189; measured epoch-body seconds: 812.33. This excludes preparation, tests and reporting.
