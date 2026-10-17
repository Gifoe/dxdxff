# Actual Raw pre-state retention experiment

## Conclusion and scope

All **15 fresh fits / 320 epochs** completed and were independently verified.
None of the five ownFITs passed the frozen first-path advancement gate. No
additional replication path, original Development fit or new Outer evaluation
was run. This is the prescribed outcome for this experiment, not a declaration
that the broader Raw-only optimization goal is achieved or impossible.

The target remains **full80, equal-patient Macro-F1 >0.65**. The following
internal assessment uses **86 appearances / 59 unique patients per arm**,
not80Outerpatients and not86independent people. Internal performance, EZ-F1,
pooled F1 and confidence upper limits cannot replace that target. The previous
complete full80 Native result remains0.630493; this study provides no new
full80 performance point.

Retaining learned pre-state alongside post-minus-pre contrast produced an
internal Macro-F1 point0.627837, versus the stronger preceding compact
PRE_MEAN point0.625949. Their paired interval crossedzero. The new model
improved MRR at the nominal interval level, but lost NEZ-F1 and ACC. Thus
there is a limited ranking signal, not a demonstrated overall F1 improvement
or a reason to override the failed advancement gate.

## Complete matched first-path matrix

All metrics are patient-equal over the same assessment appearances. AP means
average precision. The CSV `f1` field is EZ-F1, not Macro-F1. Bias is predicted
minus true EZ fraction. Every configuration and all nine endpoints are shown.

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native fixed reference | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 | -0.042319 |
| Frozen precompand reference | 0.604982 | 0.723093 | 0.535747 | 0.382907 | 0.827057 | 0.745545 | 0.746988 | 0.627907 | -0.027114 |
| Strong compact PRE_MEAN reference | 0.625949 | 0.751279 | 0.555566 | 0.408033 | 0.843866 | 0.765489 | 0.765498 | 0.662791 | -0.047233 |
| DELTA_DUP | 0.622155 | 0.746401 | 0.557058 | 0.427729 | 0.816581 | 0.739771 | 0.743852 | 0.616279 | +0.018358 |
| POST_DUP | 0.602887 | 0.730567 | 0.513294 | 0.386801 | 0.818973 | 0.737695 | 0.719459 | 0.604651 | -0.003699 |
| PRE_DELTA | 0.627837 | 0.749808 | 0.571669 | 0.426657 | 0.829017 | 0.750331 | 0.809578 | 0.720930 | +0.003843 |

The old compact PRE_MEAN is included rather than comparing only with the
weaker frozen references. Its score/checkpoint lineage and role identities
were hash-verified; it is a fixed reference, never an input to a fresh network.
It has a different head parameterization/history from the new matched arms,
so the comparison is a strong benchmark, not equal-capacity attribution.

## What information changed

A single shared CNN encodes all valid Raw windows. For each channel/seizure,
`B` is the pre-onset learned embedding mean and `P` contains the two post-phase
embedding means. Across available seizures, `D96` is the mean/populationSD
of(P-B), `P96` is the mean/SD ofP, and `B48` is the mean/SD ofB.

- DELTA_DUP presents(D96,D96) to the sole classifier.
- POST_DUP presents(P96,P96), without latent-reference subtraction.
- PRE_DELTA presents(D96,B48,B48), retaining the learned pre-state.

All three apply the same complete-patient float64 Z and use one192to64to1
head,14,577parameter positions and exactly matching random initial CNN/head
tensors per fold. Duplicate controls match tensor shape/parameter positions,
not effective information capacity. They add no distinct pre-Z statistic.
POST_DUP still uses the fixed pre-only waveform normalization: it is not an
experiment removing all pre-onset data. No physiological predictor features,
D1 branch/logits, old embeddings/weights, second classifier or identity input
enters a fresh model. Generic learned moments are not claimed novel or
interpreted as measured physiological band power.

The original valid masks, pre/0to8/8to20 supports, available-seizure handling
and full-patient Z are unchanged. All valid windows are encoded, but late
windows outside the fixed pooling support are not claimed to affect logits.
No new extraction, resampling, patient/window exclusion or Outer tuning occurs.

## Paired uncertainty: Macro-F1 and tradeoffs

These are nominal95% percentile intervals from10,000 paired patient-cluster
bootstrap draws. All repeated appearances of a patient are drawn together.
They condition on reused assessment data and fitted models, and are not
multiplicity-adjusted or independent confirmation.

| Macro-F1 difference | Point | Nominal paired95% interval |
|---|---:|---|
| DELTA_DUP - Native | +0.020181 | [-0.006691, 0.049697] |
| DELTA_DUP - frozen precompand | +0.017173 | [-0.007489, 0.044264] |
| DELTA_DUP - compact PRE | -0.003795 | [-0.017787, 0.011103] |
| POST_DUP - Native | +0.000913 | [-0.029736, 0.034457] |
| POST_DUP - frozen precompand | -0.002095 | [-0.030594, 0.028309] |
| POST_DUP - compact PRE | -0.023063 | [-0.043198, -0.002094] |
| PRE_DELTA - Native | +0.025863 | [-0.002198, 0.055798] |
| PRE_DELTA - frozen precompand | +0.022855 | [-0.002813, 0.049526] |
| PRE_DELTA - compact PRE | +0.001887 | [-0.014402, 0.018674] |
| PRE_DELTA - DELTA_DUP | +0.005682 | [-0.011012, 0.021875] |
| PRE_DELTA - POST_DUP | +0.024950 | [0.006633, 0.042919] |
| POST_DUP - DELTA_DUP | -0.019268 | [-0.040216, 0.000828] |

Keeping the pre-state does not establish a Macro gain over contrast alone.
The matched contrast-versus-post deletion is negative in point estimate, but
its interval includeszero; it is weaker than a universal reference-necessity
claim. PRE_DELTA beats POST_DUP at the nominal interval level, but the added
information/parameterization means this does not isolate one biological cause.

| PRE_DELTA minus stronger compact PRE | Difference | Nominal paired95% interval |
|---|---:|---|
| AUROC | -0.001472 | [-0.017672, 0.014230] |
| AP | +0.016103 | [-0.001738, 0.032805] |
| EZ-F1 | +0.018624 | [-0.006829, 0.045157] |
| NEZ-F1 | -0.014849 | [-0.024153, -0.005895] |
| ACC | -0.015158 | [-0.029381, -0.000671] |
| MRR | +0.044080 | [0.004385, 0.085285] |
| Top-1 | +0.058140 | [0.000000, 0.118280] |
| EZ-frac. bias | +0.051076 | [0.039244, 0.063918] |

The MRR finding is retained, but not substituted for Macro-F1. The changes in
EZ/NEZ F1 partially cancel in their Macro average. Bias is signed; a positive
difference is not automatically an improvement. Here the reference's negative
mean bias moves closer tozero, without establishing better per-patient
calibration. This does not identify a causal explanation for the F1 tradeoff.
The complete108paired and54absolute interval rows are published, not only
the positive ranking endpoint.

## Fold and source stability

Fivefold Macro-F1 sampleSD: Native0.028263, frozen precompand0.019796,
compact PRE0.026657, DELTA_DUP0.026290, POST_DUP0.034295,
PRE_DELTA0.016385. Lower spread for PRE_DELTA in these five reused internal
assessments is descriptive, not a full80 generalization or seed-robustness proof.

| Internal source Macro-F1 | HUP | LZU | Multicenter | Pediatric |
|---|---:|---:|---:|---:|
| Compact PRE | 0.657971 | 0.552926 | 0.646387 | 0.601864 |
| DELTA_DUP | 0.640103 | 0.570172 | 0.637452 | 0.623124 |
| POST_DUP | 0.623409 | 0.582550 | 0.640909 | 0.494800 |
| PRE_DELTA | 0.647971 | 0.587283 | 0.657341 | 0.574198 |

The source assessment appearances are40/19/17/10, with28/14/10/7unique
patients respectively. Gains versus compact PRE are not uniform across
sources. No center-specific model or new patient filter is chosen from these
results. The full six-arm/nine-metric source table and finite coverage are
published; subgroups with fewer than five unique people would be suppressed.

## Frozen race and no new Outer

| OwnFIT fold | Chosen first candidate | Macro gain vs strongest reference | AUROC gain vs strongest reference | Advanced? |
|---|---|---:|---:|---|
| 1 | PRE_DELTA | +0.008256 | +0.008115 | No |
| 2 | PRE_DELTA | -0.003916 | +0.006113 | No |
| 3 | DELTA_DUP | -0.008259 | +0.012674 | No |
| 4 | DELTA_DUP | +0.024063 | -0.025514 | No |
| 5 | DELTA_DUP | -0.002456 | +0.001378 | No |

First candidates use the registered source-penalized Macro criterion. The
gate requires Macro>=strongest reference+0.01 and AUROC>=strongest reference
-0.01, with compact PRE included. Fold1 misses the Macro condition; fold4
fails the AUROC condition despite a higher Macro point. The remaining
candidates fail the Macro condition. No gate was softened afterward.

Consequently, none of the additional two-path replications ran and the
stricter replicated final-eligibility rule was not exercised. Its interval
check count is explicitly0, not evidence that unrun replications passed.
The legal fallback choices are Native/Native/Native/frozen precompand/Native.
No fallback or favorable fresh point was newly evaluated on Outer just to
obtain another test table. Empty replication tables are genuinely empty.

## Actual verification and execution

The15retained CAL/HELD checkpoints were actually replayed: maximum logit
error0, all201CAL threshold decisions exact, all initial states matched across
the three fresh arms. All516patient metric rows and nine sklearn/ranking
metrics were independently checked. Public cell summaries and race decisions
were recomputed. All162complete-first-path absolute/paired CI metric rows
passed independent bootstrap reconstruction, maximum difference
9.80118763926896e-17. All nine endpoints were finite on every appearance.

Five unique changed-path tests passed, including exact original contrast
output **and gradient**, retained baseline information, missing-seizure
moments, channel ordering and maskedNaNpadding. All three full-patient GPU
backwards passed on the actual largest legal training case:164channels /
48,380validwindows, finite gradients, zero admission optimizer updates.
The complete original ownFIT role identities and stronger-reference hashes
were checked. Old science/checkpoints/seals and unrelated jobs were preserved.

Formal epochs totaled658.561329seconds, mean2.058004s/epoch, excluding
admission, HELD prediction, replay and orchestration. All retained epochs>=1.
There was no trainer restart, discarded completed epoch or loosened numerical
tolerance. The local SSH stderr-pump timed out after an empty600second stream;
the same server trainer/foreground owner remained live and progressed. This
was not a native training failure. The coordinator subsequently exited0 with
`PRESTATE_RAW_INTERNAL_PIPELINE_COMPLETE_NO_OUTER`.

Source/protocol were pushed before formal fitting at
`aad65d31c5ca682d8e56db81037223e0a12c9fea`. The scientific sources remain
byte-unchanged. Release-only source/table checks do not alter that binding.

Key SHA256 locks:

- Input binding digest: `5ad4f362e0d246cbebdd8767952669584f899df02828231134a59a893021fbad`
- Internal result seal: `a4d05159a96c6fc10aa93f21bf51fa546e7d6402754f3b85c4afb893945e784a`
- OwnFIT choices: `7af8e9df960133b6c73d59465c6b632866fb04a94b97b7f1c859812a4d4b01fd`
- Whitelisted17file delivery: `85077f2fde899461966ee4cda62019365dc6e21498962718736b80aaf9624e17`

Only source, hashes and anonymous aggregates are uploaded. Raw, identities,
truth, weights, individual predictions/latents and logs stay private. These
reused internal assessments are not independent confirmation; any eventual
historical Outer test remains exploratory repeated-test. Further work must
not choose a model, threshold, gate or patient subset from viewed Outer.
