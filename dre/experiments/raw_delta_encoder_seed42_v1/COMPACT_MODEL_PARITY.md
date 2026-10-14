# Compact DELTA parity

The historical 26 retained DELTA coordinates consist of four constant zeros, eight auxiliary duplicates and fourteen varying unique coordinates. The compact model preserves the 88-position LayerNorm mean/variance, multiplicities and zero mass; it does **not** use LayerNorm(14).

Let A_i = W_i gamma_i and Z = sum of A_i over zero positions. Folded weight for unique j is sum of A_i over its copies minus Z*m_j/n0; bias is b + W*beta. Compact normalization uses mean sum(m_j*x_j)/88 and variance [sum m_j*(x_j-mean)^2 + n0*mean^2]/88 with original epsilon 1e-5.

The trained historical D0 checkpoints were folded and tested on every legal FIT/VAL row in each fold, with exact validation decisions and threshold/checkpoint replay. Float32 logit tolerance is 2e-6, score tolerance 1e-6; synthetic float64 algebra tolerance is 1e-12, fixed before replay. Actual maximum differences are in `COMPACT_PARITY.json`. The compact classifier has exactly 1,537 trainable parameters.

E1/D1 are freshly retrained compact parameterizations, sharing the same folded scratch initializer within each fold. They are not optimizer-equivalent retraining of the original 8,817-parameter D0; E1 is the required matched optimization/compression control. D1 is never warm-started from trained E1. D0 was replayed on Development only, not retrained and not evaluated on Outer.
