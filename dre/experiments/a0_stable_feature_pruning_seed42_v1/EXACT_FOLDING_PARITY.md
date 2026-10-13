# Executed exact-folding parity

All five authoritative frozen F0 checkpoints and legal FIT/VAL banks passed.
These measurements are from this server run, not historical substituted values.

| Fold | Float64 max logit discrepancy | Float32 max logit discrepancy | Frozen probability discrepancy | CUDA probability discrepancy |
|---|---:|---:|---:|---:|
| 1 | 1.3323e-15 | 7.1526e-7 | 1.1921e-7 | 1.1921e-7 |
| 2 | 8.8818e-16 | 5.9605e-7 | 1.1921e-7 | 1.1921e-7 |
| 3 | 8.8818e-16 | 7.1526e-7 | 1.1921e-7 | 1.1921e-7 |
| 4 | 1.3323e-15 | 5.9605e-7 | 1.1921e-7 | 1.7881e-7 |
| 5 | 1.3323e-15 | 8.3447e-7 | 1.1921e-7 | 1.1921e-7 |

Threshold decisions are identical, original validation threshold selection
replays exactly, and all original patient metrics replay within1e-12. Frozen
B0 predictions remain the baseline; tiny re-evaluation probability differences
do not replace them. Full precision and source/bank/protocol hash bindings are
in `FOLDING_PARITY.json` and `DEPENDENCY_HASHES.json`.

Synthetic tests also pass in float64/float32 with nondefault learned LN affine,
zero/small/large inputs, full42 and independently pruned coordinate sets. The
normalization always reconstructs88 positions, not native LayerNorm42.
Changing any deleted or auxiliary input field has bitwise no effect on the
compact model, since only canonical retained coordinates are gathered.
Each final trained compact cell separately checks that independence on actual
validation tensors. Count is4225 trainables for full42; removing each two-field
group removes192 input weights. No trainable mask or residual input path.

Folding a trained checkpoint establishes function identity, not equivalent
training gradients, weight-decay geometry or optimizer trajectories. All new
models train from folded **untrained** seeded initialization. B1−B0 isolates
the resulting optimization/selection change; it must not be misinterpreted as
benefit from removing information that was already algebraically redundant.
