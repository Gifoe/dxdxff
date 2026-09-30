# Engineering change log

## 2026-09-30 — masked quantile pooling vectorization

The initial implementation evaluated temporal quantiles one channel at a time.
That created three CUDA quantile launches per channel per record and made the
first development epoch impractically slow. The pooling definition has not
changed: it is still the mean, Q25, Q50, Q75, and maximum over exactly the
valid windows/records. The implementation now applies NaN-masked vectorized
reductions, which exclude the same invalid entries.

The executable audit compares the vectorized output to the prior explicit-loop
rule on irregular masks and singleton support, and verifies finite gradients.
No data, feature, architecture, loss, optimizer, selection, threshold, or
split setting changed. Any pre-epoch partial computation is discarded; the
label-independent token cache and train-only scaler are retained.
