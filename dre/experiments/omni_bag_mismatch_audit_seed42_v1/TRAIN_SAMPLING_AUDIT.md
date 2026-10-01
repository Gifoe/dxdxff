# TRAIN Sampling Audit

The historical synchronized TRAIN cache stores one shared set of positions per EDF, then applies those positions to every selected channel. The reconstruction of the official extractor uses up to five 60-second windows, a 1-second edge margin, and deterministic per-EDF sampling for interruption-invariant caching. The windows can start at arbitrary samples and therefore need not be members of the non-overlapping TEST grid.

- EDFs: 296
- Maximum clips per EDF: 5
- Median extracted clips per EDF: 5.0
- P(channel bag N=5): 0.979925
- Overlap within an EDF: possible under the historical arbitrary-start rule; stored starts are authoritative.
- Model/label/split changes: none.
