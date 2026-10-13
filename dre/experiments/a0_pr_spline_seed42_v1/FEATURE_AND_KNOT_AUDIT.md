# Feature and knot audit — actual five-fold data

## Identity and provenance

Original cohort metadata checks: 80 patients, 7,635 canonical channels,
1,743 EZ and 5,892 NEZ. This is the historical contract, not the number of
independent development observations. Actual evaluation uses the original
65 validation-fold appearances from 47 unique patients. Each fold's legal
FIT/VAL bank excludes its outer patients; no global feature/label array is
indexed for new outer inference. Patient Z and FIT-only imputer/scaler replay
bitwise on all legal banks. See `BASELINE_REPRODUCTION.json` and
`PRETRAIN_AUDIT.json` for original/new bank hashes and individual check flags.

| Artifact | SHA256 |
|---|---|
| Original 88D archive | `1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd` |
| Feature order | `b29b29129d4d4a275d9cd077aa47e46cf428fe0f25913dea75a51ff0c7ea6ca9` |
| Original partition manifest | `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278` |
| Original feature-group manifest | `f7869e994e8494397761e4efa0a2a2057c2a13541331e5933e94c6f7b82a065f` |
| Registered protocol bytes | `1891ad16fe816938a977e26f23a17ab648014d50af16c0de68cfa35bc4932680` |

Names and manifest duplicate relationships, never guessed positions, determine
the 56 representatives, 16 auxiliary duplicates and 16 constant-zero columns.
B1/B3 preserve the latter and share each duplicate's main transform. Original
88D inputs remain unchanged on disk; transformation is inside the model just
before original LayerNorm. B2's prescribed generic residual can change those
dimensions, intentionally not masked or redesigned.

## FIT-only construction

Each FIT patient receives mass 1/P; every channel within that patient receives
1/(P*C). Quantiles [.02,.10,.25,.50,.75,.90,.98] use stable ascending order,
aggregate exactly equal values, normalize weights, and take the first empirical
CDF crossing (left search). No interpolation, label, VAL statistic, outer data
or epsilon jitter enters construction. Knots are frozen float64 buffers.
FIT-only and patient-channel replication tests pass. FIT means/std and empirical
knots are in `FEATURE_KNOT_AUDIT.csv`; seven values are reported by feature name.

All 280 feature/fold rows have seven distinct knots and eight active intervals:
56 active curves and 448 active stored slope parameters per fold. No real
feature triggered the fewer-than-three-boundaries identity rule. Synthetic
repeated/degenerate-knot cases pass and unused parameters are gradient-masked.
Allocated/effective parameter counts and shared initial backbone hashes are in
`PARAMETER_AUDIT.csv` (B0 8,817; B1 8,873; B2 9,259; B3 9,265).

Distinct is not equivalent to well-separated or statistically identifiable.
313 of 1,680 adjacent interior knot gaps are below 1e-7 on the preprocessed
scale. These are genuine stored input values; no fabricated jitter merges or
separates them. The registered exact-duplicate rule remains unchanged. Thus
448 is the active algebraic count, not evidence of 448 independently learned
or useful clinical degrees of freedom.

## Learned curves and input distributions

These are actual selected-checkpoint diagnostics from
`SPLINE_RESPONSE_AUDIT.csv`, not simulated results. The following averages
weight the 280 feature/fold rows equally; within-row distribution/transform
statistics pool channels. They are not the patient-equal clinical estimator.

| Diagnostic | Actual value |
|---|---:|
| Minimum / maximum learned interval slope | 0.956295 / 1.040381 |
| Mean absolute slope deviation from 1 (all intervals) | 0.007924 |
| Mean absolute deviation, two outer intervals | 0.009941 |
| Mean absolute deviation, six interior intervals | 0.007251 |
| FIT mean absolute transform (representatives) | 0.003670 |
| VAL mean absolute transform (representatives) | 0.003411 |
| FIT observations outside outer knots | 0.038742 |
| VAL observations outside outer knots | 0.033419 |
| Maximum feature/fold VAL outer fraction | 0.098651 |

The learned maps stay close to identity and far from the slope caps. Outer
slopes change somewhat more on average, but interior slopes also change; slope
size and observation fractions do not identify which region causes a metric
change. No tail-only clinical intervention was registered or run. Therefore
tail-driven benefit is **not established**, rather than inferred from nonzero
tail responses. Per-feature FIT/VAL minima, maxima, VAL medians, all knots,
slopes and transformation magnitudes remain in the CSV.

Largest VAL transformations include fold1 ABS RMS seizure mean (0.014495),
fold4 DELTA entropy seizure mean (0.013412), and ABS entropy seizure std in
fold1/fold4 (0.012884/0.012806). These are response summaries, not clinical
importance or feature-selection proposals. No feature was removed or retuned.

## Numerical conclusion

Initial input identity and original-logit parity pass for synthetic/extreme
inputs and all legal FIT/VAL banks on CPU/GPU. Zero anchor, continuity,
positive bounded derivatives, no-clipping linear tails, duplicate sharing,
constant preservation and active finite gradients pass. Independent final
buffer/slope/hash checks pass. An increasing per-feature map may still change
the final multivariate ranking after LayerNorm and the jointly trained MLP;
this distinction is measured in the frozen intervention/ranking summaries.
