# Normalization mechanism: actual frozen development diagnostics

Only original legal FIT/VAL tensors and selected frozen models are used here.
No model/threshold is reselected. Statistics do not use clinical labels;
correlations with ranking/metric changes are explicitly post-freeze descriptive
diagnostics. No formula, feature set or epsilon is selected from these findings.

## Distortion is measurable, harmful bias is not established

The upstream Patient Z centers each feature across that patient's channels.
LayerNorm instead centers different features within one channel. These are not
the same operation. Although the global patient-equal mean88 is near zero,
within-channel mean88 has SD0.405203 and 5th/95th quantiles −0.355183/0.609089.
Unique56 mean has SD0.482541 and quantiles −0.426236/0.755835. Thus dataset-wide
zero means do not make LayerNorm mean subtraction irrelevant per channel.

Let Su=sum of squared unique56 values and Sd=sum of squared auxiliary16 copies.
The raw RMS88 squared is(Su+Sd)/88, whereas RMS56 squared is Su/56. Duplication
weights selected coordinates twice in moments; the zero positions also dilute
the denominator. This algebra establishes weighting differences, not clinical
harm. For centered LN, nonzero feature means also make zero positions contribute
16*mean88^2 to the centered sum of squares.

All table distributions use equal patient-fold mass, with equal channel mass
within each patient-fold. Quantiles are inverse weighted empirical CDF, no
interpolation. Repeated FIT/VAL appearances are not treated as unique patients.
Finite fractions are reported separately; any undefined raw0/0 statistic is
not replaced by a fabricated value. These are channel-distribution diagnostics,
not the primary mean of clinical per-patient AUROCs.

| VAL diagnostic | Weighted mean | Weighted SD | 5th percentile | 95th percentile |
|---|---:|---:|---:|---:|
| RMS88 | 0.591533 | 0.569036 | 0.219554 | 1.549295 |
| RMS56 | 0.654935 | 0.628615 | 0.246964 | 1.698805 |
| RMS56/RMS88, epsilon included | 1.111573 | 0.051206 | 1.024994 | 1.193672 |
| RMS56/RMS88, raw | 1.111582 | 0.051211 | 1.024995 | 1.193678 |
| var56/var88, epsilon included | 1.181547 | 0.135150 | 0.966497 | 1.407057 |
| Duplicate raw squared-energy fraction | 0.212031 | 0.072225 | 0.093266 | 0.331426 |
| Duplicate centered LN-energy fraction | 0.188054 | 0.075325 | 0.074812 | 0.317570 |
| Zero-coordinate centered LN-energy fraction | 0.051706 | 0.059771 | 0.000323 | 0.184813 |
| RMS72/RMS88, excluding zeros only, epsilon included | 1.105534 | 0.000008 | 1.105521 | 1.105541 |

Removing16 zeros alone produces an almost constant RMS factor, approximately
sqrt(88/72)=1.105542 before epsilon. Removing duplicate weighting as well adds
channel variability; RMS56/RMS88 is not exactly constant, though its dominant
effect is a roughly11% common rescaling and the relative SD is only about4.6%.
Within-patient ratio SD averages0.046331. No duplicated coordinates were removed
from the classifier. The counterfactual72 statistics are arithmetic audits, not
a fifth model or alternative predictor.

Center-level VAL RMS ratios (mean / SD): HUP 1.115934/0.046116,
LZU 1.114620/0.056326, Multicenter 1.110951/0.052678,
Pediatric 1.098414/0.052601. FIT overall ratio is 1.115746/0.052240.
Full role/fold/center distributions are in NORMALIZATION_STATISTICS.csv and
DUPLICATE_INFLUENCE.csv, including centered means, variances and inverse factors.

The prelocked extreme-factor condition is inverse factor>10. VAL LN88 and
RMS88 exceed it with weighted frequency0.000145 (0.0145%); LN56/RMS56 frequency
is0. FIT LN88 frequency0.000361; FIT RMS56 frequency0. No patient is excluded
or input clipped due to these diagnostics. Different output magnitude is not
proof that an old normalization was wrong or a new one is clinically useful.

## Output, affine and gradients

Selected models, eval mode; per-channel input derivative is the L2 norm of the
gradient of that channel's NEZ logit with respect to its88 pre-norm inputs.
The first linear activation is measured before GELU. Gradients are label-free,
not loss gradients, optimization changes or causal explanations.

| VAL equal-patient distribution mean | B0 | B1 RMS88 | B2 Unique-LN | B3 PURN |
|---|---:|---:|---:|---:|
| Pre-input RMS | 0.591533 | 0.591533 | 0.591533 | 0.591533 |
| Post-normalization RMS | 0.999712 | 1.000401 | 0.938406 | 0.902433 |
| Input derivative L2 | 0.884244 | 0.803269 | 1.275619 | 0.768368 |
| First-linear activation L2 | 6.499434 | 6.486426 | 6.648977 | 5.963111 |
| Mean absolute normalized zero-coordinate output | 0.442179 | 0.005198 | 0.499662 | 0.004919 |

All schemes retain88 independent gamma/beta coordinates. Across five selected
folds, gamma/beta ranges are: B0[0.962557,1.046064]/[−0.024379,0.029788];
B1[0.941095,1.085629]/[−0.021207,0.043065];
B2[0.948387,1.108660]/[−0.032146,0.035999];
B3[0.941583,1.067652]/[−0.023325,0.044998]. Their per-fold distributions are
in AFFINE_PARAMETER_AUDIT.csv. Actual selected eval-mode FIT/VAL weighted BCE
is in TRAINING_EFFICIENCY.csv; it did not select checkpoints.

Zeros remain exactly zero in original inputs. Under LN/Unique-LN, subtracting
the channel's feature mean makes those positions nonzero even before affine.
Under RMS88/PURN, they remain zero before affine beta, which may learn nonzero
values. They are not artificially reset afterward. Equal raw duplicate values
remain separately available at88 classifier positions; their affine parameters
may learn different values, just as in original A0.

## Ranking relationships and factorial attribution

Mean within-patient pair-order changes versus independently trained B0:
B1 0.080092, B2 0.056085, B3 0.080918. Ties count as ordering-state changes.
Both normalization changes and retrained downstream weights contribute; these
comparisons cannot attribute a ranking change solely to one numerical factor.

B3 correlations across65 repeated validation appearances:

| Predictor / outcome | Pearson | Spearman |
|---|---:|---:|
| Mean RMS ratio / pair-order change | −0.266668 | −0.270760 |
| Within-patient RMS-ratio SD / pair-order change | 0.091535 | 0.141696 |
| Mean RMS ratio / AUROC delta | −0.185960 | −0.189433 |
| Within-patient RMS-ratio SD / AUROC delta | 0.227269 | 0.118604 |
| Mean absolute mean shift / AUROC delta | −0.165277 | −0.065869 |

These weak/modest associations are descriptive, with no significance or causal
claim, multiple examined pairs and repeated cohort use. The full file also
contains AP relationships and the separate47-ID-mean correlation estimator.
No new architecture is selected from them.

The independently trained2x2 AUROC interaction is+0.004644,95% descriptive
CI[−0.005923,0.017437]. AP interaction+0.005666[−0.009835,0.022594];
Macro-F1−0.000074[−0.010260,0.011033]. Positive AUROC/AP point interactions
do not establish complementary physiological effects. PURN's development
increment over RMS88 is only0.003519, below the required0.005; the full gate
fails. Subsequently authorized outer results further fail to support a useful
normalization-statistics correction (see OUTER_REPORT.md). Keep original A0.
