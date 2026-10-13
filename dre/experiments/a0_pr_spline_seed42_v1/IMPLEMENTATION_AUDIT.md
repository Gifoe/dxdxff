# Implementation and pretraining admission

The exact source-free baseline comes from the original D2 ablation factory.
Its saved checkpoints are replayed against original legal FIT/VAL banks;
original dependency, feature/order/partition/checkpoint/bank hashes gate access.
The monolithic NPZ is indexed only for unlabeled identity/order/split metadata,
not global x/y. Existing fold banks contain only legal FIT/VAL patients and
exclude that fold's outer patients. No new outer prediction/result is accessed.

`model.py` instantiates the original SourceA0(seed).network, dropping the unused
source parameters rather than reconstructing its MLP. Extra modules are added
afterward. The original trainer reseeds after construction, preserving dropout
and patient-update RNG behavior across differing extra initializations.
All backbone components remain jointly trainable, including original LayerNorm.

Curve arithmetic uses float64 knots/slopes/integration before converting to the
original input dtype. Identity initialization remains bitwise input identity.
Seven patient-equal inverse-empirical-CDF quantiles are stable-sorted and exact
ties aggregated; no jitter or validation fitting. Fewer than3 distinct knots
means identity/inactive.56x8 theta storage remains fixed; unused intervals are
masked and have zero gradients. Auxiliary columns share their main's curve.

Tanh slope bounds are mathematically open(0.25,1.75). Floating-point saturation
can reach endpoints for extreme theta, without losing positive slope; actual
trained values and sampled monotonicity are checked independently. There is
no input clipping or separate output predictor. Learned individual feature
monotonicity does not imply monotonic final multivariate ranking.

Local and server synthetic contracts cover FIT independence, patient-channel
replication invariance, identity, zero anchoring, analytic positive derivatives,
continuity/linear tails, exact duplicate sharing, constant preservation,
inactive/repeated-knot behavior and nonzero active gradients. The gradient
probe deliberately covers both tails; a tiny synthetic FIT set can otherwise
have no sample beyond its boundary quantile. That test fixture correction
does not change model/data/science. No formal outcome existed at that point.

293 server pretraining checks passed: exact allocated parameter counts,
original shared initialization, CPU/GPU logits on random/extreme/each actual
FIT/VAL bank, finite gradients, FIT-only buffers and exact resumed weights,
buffers, Adam slots, RNG, history and selection for all three new arms.

The first prepare attempt used a wrong manifest **directory**, stopping before
any knot bank or new training. The actual original manifest is in the D2
runtime public directory, with the exact registered SHA. Only source location
was corrected, preserving the first code snapshot and baseline artifacts.
No input, threshold, model or criterion was changed.

All15 formal train/validation selections freeze before aggregate comparison
and interventions. Each arm uses its own original Macro-F1/EZ-F1/earlier-epoch
selection. Timing includes original training/validation/checkpoint writes;
selected FIT/VAL BCE is an eval-mode diagnostic, not a selection criterion.
Private predictions and each frozen intervention seal with hashes. Independent
sklearn metrics, alternate multiplicity-weighted bootstrap and aggregate/gate
parity checks precede final claims. Individual identifiers, arrays, clinical
records, checkpoints and logs never enter public delivery.

## Actual execution and independent verification

All 15 new clinical cells completed in a single formal attempt: 120 completed
epochs, 7--9 per cell under the original stopping rule, with 15 selected
checkpoints plus the 5 replayed historical baselines. The sum of measured
trainer invocations is 13.128539 seconds; this excludes preparation, numerical
tests, baseline replay, final bootstrap/verification, network and reporting.
The frozen features are already in memory; no signal extraction was performed.
No completed clinical epoch was discarded, no formal native failure occurred,
and training/evaluation/verification stderr files are each zero bytes. There
are no active experiment processes at completion. Synthetic deterministic
resume tests are not extra clinical arms and are not included in that timing.

`INDEPENDENT_VERIFICATION.json` records 390 patient-appearance/variant metric
cells, 112 verified paired intervals, maximum metric drift
2.220446049250313e-16, exact parameter/buffer/threshold/source binds, and PASS.
It recomputes metrics from sealed development predictions, not model inference
or threshold selection. The alternate bootstrap uses identity multiplicities,
not an independent resampling of repeated appearances.

The final development freeze SHA256 is
`5590218716c36f226791d4db08798ee8cc68be80b64aa3a85c974d7124507a92`.
No code/test/protocol changed after formal admission. The failure is a scientific
utility gate failure, not baseline corruption or incomplete execution.
