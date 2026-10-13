# Exact structural compression

The authoritative feature manifest, not guessed coordinate positions, defines
the mapping. Its hash is locked in the protocol. The admission routine verifies
all five legal FIT/VAL banks against frozen F0: original feature order, patient
and channel alignment, labels, source indices, preprocessing and masks.

Original 88-position interface: 42 unique nonconstant main coordinates,
8 exact DELTA auxiliary copies, 12 retained constant-zero band coordinates,
26 masked ZDELTA positions. Thus 50 positions carry values after multiplicity;
38 positions are zero. The seven varying physiological types each have three
views and mean/std aggregation, giving 21 two-coordinate groups.

For retained set S with multiplicities m, M=sum(m), n0=88-M:

`mu=sum(m*x)/88`

`var=(sum(m*(x-mu)^2)+n0*mu^2)/88`

`u=(x-mu)/sqrt(var+1e-5)`

Let original first-linear columns times original LN gamma be A_i. Let
Z=sum(A_i) over all zero positions. The zero-position normalized input obeys
`u0=-sum(m*u)/n0`, so folded weight column j is
`sum(A_i for original copies of j)-Z*m_j/n0`.
Folded bias is `original_bias+original_W@original_LN_beta`.
This includes all original zero and newly deleted positions; omitting Z would
not be the original model. Epsilon and denominator88 remain unchanged.

The compact trainable model has Linear(k,96), GELU, Dropout0.15, Linear(96,1):
`96*k+193` parameters. At k42 this is4225, versus8817 original. No extra
trainable LN affine, duplicate weights or mask-only deleted pathway survives.

Initializer folding is from the original **seeded untrained** SourceA0 shared
network. Trained F0 checkpoints are used only for identity testing and B0 replay.
AdamW in folded coordinates, clipping and parameter-wise decay need not yield
the original optimization trajectory; B1−B0 measures that difference, not
physiological information removal.

The frozen cache's delta/theta zero positions indicate unavailable varying
numeric information in this representation, not absent clinical low-frequency
physiology. DELTA the view is distinct from delta the frequency band. No raw
waveform re-extraction or feature alteration is performed.
