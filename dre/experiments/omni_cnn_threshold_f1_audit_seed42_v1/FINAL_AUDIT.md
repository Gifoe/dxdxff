# Frozen Omni CNN threshold and F1 audit (seed 42)

The frozen CNN reproduces channel AUROC **0.798767**,
close to the published 0.8061, independently of any threshold. Its
official-style test-derived Youden Macro-F1 is **0.599267**,
below the published 0.6469, whereas fixed 0.5 yields **0.659754**.
This is a **post-hoc, exploratory repeated-test audit**, not a new blind
benchmark or a reversal of `RAW_ENCODER_REPRODUCTION_FAILED`.

| Operating point | Threshold | Macro-F1 | Pathological F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official test-Youden | 0.016663 | 0.599267 | 0.354100 | 557 | 1782 | 5515 | 250 |
| Fixed 0.5 | 0.500000 | 0.659754 | 0.380427 | 276 | 368 | 6929 | 531 |
| Macro-F1 oracle (diagnostic only) | 0.540771 | 0.663645 | 0.385439 | 270 | 324 | 6973 | 537 |

The post-hoc maximum Macro-F1 is **0.663645** at threshold
**0.540771** (smallest threshold among 1 tied
maximizers). The Youden-to-oracle threshold separation is
**0.524108**. No oracle threshold was used
for formal performance or inside the primary bootstrap.

Youden maximizes sensitivity + specificity - 1, not Macro-F1 or precision.
At Youden versus fixed 0.5, FP changes by **+1414** and FN by
**-281**. Thus the confusion-matrix tradeoff, in the presence of
807 pathological and
7297 normal pairs, explains the F1 difference.

The 99% F1-max candidate range is [0.283875,
0.591366] (width 0.307492); the 95%
range is [0.076544, 0.806962]
(width 0.730418). Fixed 0.5 qualifies for the 99% range:
**True**; Youden qualifies: **False**.
See `F1_ROBUSTNESS.json` for every qualifying candidate and its nearest-score
distance; range hulls may include gaps.

Patient-cluster bootstrap used 10,000 seed-42
resamples of patients, including all their EDF-channel pairs, with both
thresholds held numerically fixed. The paired Macro-F1 difference
`F1(0.5) - F1(Youden)` is **0.060486** on
the full cohort, with bootstrap percentile 95% CI
**[0.010071, 0.110477]** and Pr(diff > 0) **0.990200**.
The full fixed-threshold distributions for Macro-F1, pathological F1,
sensitivity, and specificity are in
`PATIENT_BOOTSTRAP_FIXED_THRESHOLDS.csv`.

The optional time-resolved ictal audit was **not run**: no compatible frozen
time-bin scores were established for this CNN with the same channel labels,
and constructing them would be a new experiment. No per-time-bin threshold
was selected.

All 8,104 `(EDF, channel)` predictions and the 240,074 segment probabilities
remain **private on the original server**. Replayed scores matched the prior
aggregate AUROC, both Macro-F1s and Youden threshold to the locked tolerance
before the score table was persisted. No checkpoint, waveform cache, labels,
or predictions were modified. The frozen table SHA-256 is
`d8a500a578c613d177e6513c8692040ab25d5fe0da64fba22631b7b9ae4eaa14`.

The evidence supports a specific operating-point explanation for the F1
gap in this frozen model, **not** a claim that the published Youden Macro-F1
was reproduced or that A1-NET's hard gate passed. Published-code extraction
frequency ambiguity remains a separate provenance limitation. The user's
question 13 was truncated in the supplied message; no additional unprovided
claim is inferred here.

## Direct answers to the supplied questions

1. **AUROC independently reproduced?** Yes: 0.798767
   versus published 0.8061; AUROC uses score ranking, not a fixed threshold.
2. **Reproduced Youden threshold:** 0.016663.
3. **Macro-F1 at Youden:** 0.599267.
4. **Macro-F1 at 0.5:** 0.659754.
5. **Post-hoc maximum Macro-F1:** 0.663645; diagnostic only.
6. **Macro-F1-maximizing threshold:** 0.540771 (smallest tied candidate).
7. **Threshold separation:** 0.524108.
8. **Why Youden selects that point:** Its J is 0.446001, versus
   0.291576 at 0.5; the rule rewards sensitivity and specificity
   equally, irrespective of precision and class imbalance.
9. **Error terms behind its F1 loss versus 0.5:** FP 1782 versus
   368 (+1414); FN 250 versus 531
   (-281). The full TP/TN counts are in the table above.
10. **Is 0.5 inside the high-F1 region?** 99%: True;
    95%: True (candidate-level membership).
11. **Is Youden outside it?** 99% outside: True;
    95% outside: True.
12. **Is the 0.5-versus-Youden difference stable across patients?**
    Paired 95% interval [0.010071, 0.110477]; strictly above zero:
    True. This is a fixed-threshold resampling audit, not an
    independent test set.
13. The supplied question 13 ends after an opening backtick, so its proposed
    assertion cannot be evaluated without inventing text.
