# Magnitude identifiability audit

The mandatory pretraining checks passed on all five legal FIT/VAL banks;
no outer features or labels were materialized. The two statistics were
defined before observing results, fitted label-free, and were nonconstant.

## FIT-only geometry

| Statistic | Patient-equal FIT mean range | FIT std range | Within-patient variance range | Between-patient variance range |
|---|---:|---:|---:|---:|
| Channel mean of88D input | approximately zero (absolute value below5.4e-11) | 0.44024–0.46117 | 0.19381–0.21268 | 7.3e-20–1.1e-19 |
| Channel log std | −0.87929 to−0.79329 | 0.57982–0.60854 | 0.28402–0.30321 | 0.05217–0.06864 |

Both have nontrivial **within-patient channel variation**. The channel-mean
statistic does not behave like a varying patient-level mean: prior Patient Z
makes its between-patient mean variation effectively zero. The log-std
statistic has variation both within and between patients. No nonfinite or
degenerate channel statistics occurred. Weighted quantiles and extremes
are in `results/MAGNITUDE_STATISTICS.csv`.

After the original preprocessing, feature extremes range from roughly−40.01
to68.36 across folds. Column FIT std ranges from0 (the retained constants)
to approximately1.000000034. The exact16 duplicate auxiliary dimensions
and16 constant-zero dimensions are retained and tested. Duplicates give some
physiological coordinates extra representation weight; zero dimensions change
the88D mean/variance denominator. These statistics cannot be called independent
physiological magnitudes, nor do they recover raw EEG amplitude removed by
Patient Z/FIT scaling.

## LayerNorm scale diagnostic

For fixed synthetic vectors, positive rescaling by0.25,0.5,1,2,4 changes
input norm by exactly the same factors. Under each frozen B0's learned
LayerNorm, maximum output drift across these tests is0.000336647 and maximum
NEZ-logit drift is0.000078976. LayerNorm is approximately, not perfectly,
positive-scale invariant because epsilon is1e-5. The mathematical removal
of most input shift/scale information does **not** demonstrate that the
removed information predicts clinical labels or should be restored.

## What the trained intervention does establish

Correct rather than within-patient shuffled statistic correspondence yields
development AUROC +0.004347 [0.001130,0.007889], AUPRC +0.023887
[0.008984,0.041371], Macro-F1 +0.019760 [0.010309,0.030894]. These are
fixed-checkpoint/fixed-threshold paired patient-cluster intervals. Shuffling
preserves the statistic multiset, input and original hidden representation.
The trained PMA therefore depends on channel-statistic alignment.

It does **not** establish a useful incremental localization model: versus
independently trained A0, AUROC gain is only0.000807 with an interval crossing
zero; versus equal-capacity B2 it is0.001193, also with an interval crossing
zero. Simple removal of LayerNorm performs materially better than PMA in
this development comparison. In the secondary47-ID-equal aggregation PMA
AUROC is0.721521 versus A0's0.721746, reinforcing that the small primary
point difference is not robust to the declared reporting unit.

Decision: **magnitude correspondence is used by this trained model, but useful
incremental prediction and a mechanism-specific improvement over ordinary
capacity are not established.** No outer validation is authorized.
