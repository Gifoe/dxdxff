# Distributional feature definition

For one frozen `(EDF, channel)` segment-probability vector `p`:

- `mean_score = mean(p)`
- `K = max(1, ceil(0.20 * len(p)))`
- `tail_excess = mean(top-K(p)) - mean_score`
- `q90_excess = quantile(p, 0.90, method="linear") - mean_score`
- `heterogeneity = std(p, ddof=0)`

Each OOF fold fits median and IQR independently on fold-train patients. Transformation is `(x - median)/(IQR + 1e-6)`, clipped to `[-5, 5]`.

The adapter has no intercept and exactly three possible coefficients. At all-zero coefficients, output equals `mean_score` to floating-point precision.
