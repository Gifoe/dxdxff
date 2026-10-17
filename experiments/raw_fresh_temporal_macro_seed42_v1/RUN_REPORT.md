# Actual fresh Raw temporal-interaction experiment

## Decision

**No fresh candidate passed final internal eligibility. No new original
Development fit or Outer evaluation was run.** All21freshfits/555epochs,
including the six prescribed fold4 replication fits, completed and passed
independent verification. The overall full80 patient-equal Macro-F1>0.65 goal
remains unachieved; completing this negative experiment is not completing it.

The tested temporal residual did not improve Macro-F1. Its complete first-path
point was0.605796 versus0.627126for the fresh base and0.624601for the equal-
parameter pointwise control. The nominal paired intervals favor both controls.
The pointwise control's positive fold4 first-path signal did not survive the
registered confidence/path-consistency gate, despite positive replicated mean
differences. No gate was relaxed after seeing those outcomes.

These are reused internal assessment results: **86 patient-assessment
appearances /59unique people per complete-first-path arm**. Every appearance
has equal weight, so this is not an equally weighted59-person mean, nor a
new80-patient Outer result. Repeated appearances of each person stay together
in bootstrap draws. Internal data have also been reused across studies; nominal
intervals are not independent confirmation or multiplicity-adjusted guarantees.

## Complete first-path results

All seven arms and all nine endpoints are retained. AP is average precision;
the CSV `f1` field denotes EZ-F1. Bias is predicted minus true EZ fraction.

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Native | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 | -0.042319 |
| Fixed precompand | 0.604982 | 0.723093 | 0.535747 | 0.382907 | 0.827057 | 0.745545 | 0.746988 | 0.627907 | -0.027114 |
| Prior PRE_MEAN | 0.625949 | 0.751279 | 0.555566 | 0.408033 | 0.843866 | 0.765489 | 0.765498 | 0.662791 | -0.047233 |
| Prior retained pre-state | 0.627837 | 0.749808 | 0.571669 | 0.426657 | 0.829017 | 0.750331 | 0.809578 | 0.720930 | +0.003843 |
| FRESH_BASE | 0.627126 | 0.760838 | 0.572702 | 0.439298 | 0.814953 | 0.737752 | 0.780884 | 0.686047 | +0.027709 |
| FRESH_POINTWISE | 0.624601 | 0.762165 | 0.573373 | 0.440449 | 0.808753 | 0.732529 | 0.787021 | 0.686047 | +0.042954 |
| FRESH_TEMPORAL | 0.605796 | 0.750109 | 0.555579 | 0.401047 | 0.810544 | 0.728416 | 0.770522 | 0.674419 | +0.016056 |

The stronger prior Raw controls are included, not only the weaker fixed Native
and precompand references. Their scores/thresholds and exact role/archive
identities were hash-verified, not newly fitted or passed into fresh networks.
Their parameterization and training history differ: these are performance
references, not matched initial-state attribution controls.

The highest fresh Macro point is essentially the preceding retained-pre-state
point, not an established improvement. No internal point or favorable endpoint
is substituted for full80Macro-F1>0.65. The old complete80patient Native value
remains0.630493; none of the fresh models has a new full80 metric in this study.

## Paired Macro-F1 uncertainty

Nominal95% percentile intervals from10,000 patient-cluster draws, seed42.
The complete63absolute and135paired interval rows are published.

| Contrast | Difference | Nominal paired95% interval |
|---|---:|---|
| BASE - Native | +0.025152 | [-0.001457, 0.053860] |
| BASE - fixed precompand | +0.022143 | [-0.001818, 0.048709] |
| BASE - prior PRE_MEAN | +0.001176 | [-0.014891, 0.017965] |
| BASE - prior pre-state | -0.000711 | [-0.016029, 0.015017] |
| POINTWISE - Native | +0.022627 | [-0.002676, 0.051133] |
| POINTWISE - fixed precompand | +0.019618 | [-0.003184, 0.045769] |
| POINTWISE - prior PRE_MEAN | -0.001349 | [-0.018575, 0.016393] |
| POINTWISE - prior pre-state | -0.003236 | [-0.019669, 0.013854] |
| TEMPORAL - Native | +0.003822 | [-0.020310, 0.030118] |
| TEMPORAL - fixed precompand | +0.000814 | [-0.021179, 0.024796] |
| TEMPORAL - prior PRE_MEAN | -0.020154 | [-0.033937, -0.005737] |
| TEMPORAL - prior pre-state | -0.022041 | [-0.040101, -0.005046] |
| TEMPORAL - BASE | -0.021330 | [-0.033699, -0.011213] |
| TEMPORAL - POINTWISE | -0.018805 | [-0.029976, -0.009080] |
| POINTWISE - BASE | -0.002525 | [-0.010915, 0.004122] |

The evidence rejects a useful Macro gain for this tested temporal adapter under
this budget/objective, not temporal modeling in general. A larger local
interaction path alone was not sufficient. Equal parameter counts do not make
the modules' function classes identical: the pointwise block has a20-channel
bottleneck and output bias, whereas the temporal block uses24depthwise filters
and no output bias. The result cannot isolate every difference as purely time.

## What was actually compared

One shared Raw CNN, one waveform normalization and one96to64GELUto1classifier
are used in every fresh arm. The base has8,433parameters. Two controls add
984parameters each (9,417total), returning
`mean(x) + 0.25*mean(tanh(adapter(x)))` before patient aggregation:

- POINTWISE:24to20,k1/GELU/20to24,k1; zero output projection/bias initially.
- TEMPORAL:24depthwise,k17,padding8/GELU/24to24,k1; center-identity depthwise
  initialization, zero output projection initially.

All shared CNN/head tensors and initial logits match exactly across all three
in each fold. Complete adapter state tensors differ in shape and are not
claimed identical. Every model starts from newly sampled, matched core weights;
no old trained state is supplied. Epoch0cannot be selected. All parameters
remain trainable with the same Macro-oriented objective, optimizer and budget.

The longest local nonlinear path before global pooling is33input samples for
BASE/POINTWISE and289for TEMPORAL. The base still pools every latent temporal
position, so it never received only33samples in total. This is generic neural
architecture, not a novel physiological measurement or an interpretable fixed
frequency-band analysis. No hand-engineered physiological predictor, D1 branch,
old logit/embedding, identity feature or second classifier is supplied.

Pre-only common-scale waveform asinh normalization, pre-latent subtraction,
0to8/8to20phase means, available-seizure mean/populationSD, original masks and
full-patient float64 Z stay unchanged. Every legal included patient's valid
Raw windows are forwarded; encoded late windows are not falsely claimed as
classifier support. No new extraction, resampling or quality-based removal
occurs. The frozen original80patient cohort is preserved, but this internal
experiment is not a new full80 waveform census or test evaluation.

Earlier negative experiments are not hidden: the prior RF535 scratch trial
was EZ-oriented, and the previous matched984parameter temporal trial warmed
trained two-waveform bodies with eligibleepoch0. This study removes those
specific differences but still fails to establish a useful temporal gain.

## Prespecified race and actual replication

| OwnFIT fold | First candidate | Macro gain vs strongest reference | AUROC gain vs strongest reference | Advanced? |
|---|---|---:|---:|---|
| 1 | BASE | -0.041623 | -0.018274 | No |
| 2 | POINTWISE | +0.007310 | +0.013534 | No |
| 3 | BASE | -0.009904 | +0.006527 | No |
| 4 | POINTWISE | +0.022266 | +0.009089 | Yes |
| 5 | TEMPORAL | -0.008938 | +0.008450 | No |

Selection uses the fixed ownFIT source-penalized Macro score, then parameter
count/order. Advancement requires>=0.01Macro gain against all four fixed
references and AUROC>=their strongest-0.01. Fold2's positive point does not
meet that Macro condition. Fold4 alone advances, and **all three** fresh arms
are replicated on two additional paths, not just the first-path winner.

The three held-out paths of this ownFIT cover52assessment observations of
52unique patients. These are not52newOuterpatients. The complete replication
Macro means were Native0.609564, fixed precompand0.618122, prior PRE_MEAN
0.621822, BASE0.634024, POINTWISE0.635451 and TEMPORAL0.621624.
PRIOR_PRESTATE never ran those historical extra paths, so no fictitious
replication row is added for it.

| Fold4 replication Macro contrast | Difference | Nominal paired95% interval | Path wins |
|---|---:|---|---:|
| BASE - Native | +0.024460 | [-0.003009, 0.051734] | 2/3 |
| BASE - fixed precompand | +0.015902 | [-0.008824, 0.041121] | 1/3 |
| POINTWISE - Native | +0.025887 | [-0.001739, 0.053381] | 2/3 |
| POINTWISE - fixed precompand | +0.017329 | [-0.007733, 0.042682] | 1/3 |
| TEMPORAL - Native | +0.012059 | [-0.014383, 0.038399] | 2/3 |
| TEMPORAL - fixed precompand | +0.003502 | [-0.021570, 0.027824] | 1/3 |

Means remain positive for several contrasts, but the intervals crosszero and
none wins at least2/3paths against fixed precompand. TEMPORAL also misses the
>=0.015mean-gain requirement. Thus all fresh candidates fail the frozen
replicated gate; this is insufficient reliability, not proof every positive
point is spurious. The legal fixed fallback choices are
PRE_MEAN/Native/PRE_MEAN/PRE_MEAN/PRE_MEAN, chosen only from their same ownFIT
first paths. No newly recombined fallback Outer metric is computed merely
because its historical score files are available.

## Stability and source scope

Fivefold first-path Macro sampleSD: Native0.028263, fixed precompand0.019796,
prior PRE_MEAN0.026657, prior pre-state0.016385, BASE0.020403,
POINTWISE0.017996, TEMPORAL0.021044. Similar or lower spread without a better
mean does not establish success or seed robustness.

| Source Macro-F1 | HUP | LZU | Multicenter | Pediatric |
|---|---:|---:|---:|---:|
| Native | 0.645560 | 0.520457 | 0.598747 | 0.587998 |
| Fixed precompand | 0.649986 | 0.529582 | 0.594113 | 0.586707 |
| Prior PRE_MEAN | 0.657971 | 0.552926 | 0.646387 | 0.601864 |
| Prior pre-state | 0.647971 | 0.587283 | 0.657341 | 0.574198 |
| BASE | 0.652407 | 0.584032 | 0.639424 | 0.586972 |
| POINTWISE | 0.653672 | 0.587295 | 0.622268 | 0.583162 |
| TEMPORAL | 0.635842 | 0.560826 | 0.611522 | 0.561321 |

Source appearance counts40/19/17/10 correspond to28/14/10/7unique people.
The fresh temporal arm's mean is below BASE in all four reported source
groups, but these are small reused strata, not four independent replications
or biological causal findings. No source-specific architecture or patient
exclusion is selected from them. All nine source endpoints and finite coverage
are retained; groups with fewer than five unique patients would be suppressed.

## Verification and execution

All21retained CAL/HELD checkpoints were actually replayed, maximum logit
error0. All201CAL threshold decisions matched exactly. Shared CNN/head initial
states matched per fold; complete per-arm initialization hashes were also
checked individually. All806private patient-metric rows/9metrics and public
cell summaries were independently checked. All198complete-first-path CI
metric rows passed separate bootstrap reconstruction, maximum difference
9.974659986866641e-17. The six replication-gate intervals also passed separate
reconstruction, maximum error2.0816681711721685e-17. No interval was filled
from an unrun model or a simplified proxy.

Five unique changed-path tests passed. Largest complete legal training-patient
GPU admission used164channels/48,380validwindows, with exact initial logits,
finite all-parameter gradients, nonzero CNN/head and new output-projection
gradients, and zero optimizer updates. No source/data gate or tolerance was
relaxed. All21fits/555epochs used the same uninterrupted coordinator; no
completed epoch was discarded. Epoch seconds summed1,282.794118, mean2.311341s,
excluding admission, held-out inference, replay and orchestration; this is
not a whole-pipeline speedup measurement.

The local empty-stderr SSH pump timed out after600seconds, but the same live
server worker and owner continued. No model restart followed that observation
failure. The coordinator eventually exited0 with
`FRESH_TEMPORAL_RAW_INTERNAL_PIPELINE_COMPLETE_NO_OUTER`.

Sources/protocol were pushed before formal fitting at
`f266f04b224463ff02a34fdcaaf568d055ede568`; registered scientific sources
remain byte-unchanged. Legacy mixed-file metadata readers do not provide
physical file isolation; no original Development/Outer score or label is
used for this program's fitting, model selection or new evaluation.

SHA256 locks:

- Input binding digest: `4c23e5bee03909e7e5923965db373405bdb93122c270abb5fd6c78d0649ffa0d`
- Internal result seal: `7ab66d2b6bdbb9826bc57fa67bbc4373b6fd76f1401ddbeb31a4120b8cc19b7e`
- OwnFIT choices: `454ca6c52f32adb44000f01564eb71904a255d3c16ce9c79f2d61ef726c0cc0f`
- Whitelisted17file delivery: `62aabe15d25bd86343a946ac624c0ed23a851ecd5673a4d5e8ad7b2b2d657aed`

Only code, hashes and anonymous aggregate outcomes are uploaded. Raw, truth,
identities, weights, individual scores/latents and runtime logs remain private.
No new Outer inference or best-Outer adoption occurs. The >0.65full80 target
requires an actual verified full-cohort result, not this internal experiment's
completion, its highest point, a different F1 definition or an interval bound.
