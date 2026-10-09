# Temporal CoTAR source audit, before development training

Scope: train T1 and T2 only. T0 is frozen matched A0 inference, not retraining.
No T3, raw waveform input, SSL, clinical label correction or new outer testing.

Baseline: previous PLLR branch reproduces the 88D/8,817-parameter A0 development
Macro-F1 .63807978285 (65 appearances/47 IDs). Its historical B0 outer .616167
belongs to a different scope. Reuse hash-locked PR-UAS network, preprocessing,
threshold/metric and RNG helpers; do not reuse soft targets or MC probabilities
as A0 logits. Existing 20 patient-held-out hard-label A0 teacher checkpoints may
be reused only after checking their original plan, FIT-only preprocessing,
checkpoint hashes and query exclusion from both training and selection. Generate
deterministic eval-mode logits, not logits of MC-averaged probabilities.

HLV: matched R0/R1 validation .650626/.651611, +.000985, gate failed; R0 is a
different window model, not A0. E3/E1 raw-model development F1 .616545/.613295,
SSL delta +.003251 CI[-.018788,.026011], gate failed. Those raw experiments used
an approved 255-record crop; this feature task retains all256 original records.
These results neither validate nor preclude this controlled temporal experiment.

Official TeCh inspected at commit `9a378cc546a5d97c871eff282148175b3c7cd75b`:

- [Transformer_EncDec.py](https://github.com/Levi-Ackman/TeCh/blob/9a378cc546a5d97c871eff282148175b3c7cd75b/layers/Transformer_EncDec.py)
  defines four linear projections, GELU, dimension-wise temporal softmax,
  weighted core aggregation and redistribution to each token. EncoderLayer has
  post-residual LayerNorm and a 2*d_model FFN with dropout.
- [TeCh.py](https://github.com/Levi-Ackman/TeCh/blob/9a378cc546a5d97c871eff282148175b3c7cd75b/models/TeCh.py)
  has array-index positions, optional cross-channel patching and sample-level
  global averaging/classification. None of those are appropriate copies here.

Borrow only the CoTAR algebra and EncoderLayer topology. T1 uses its own local
projected core; T2 uses the masked dimension-wise softmax aggregate. Their
parameter tensors/topology are otherwise identical. Adaptations: continuous
time-in-seconds sin/cos encoding with a shared learned projection, explicit
missing-token masks including all-missing safety, stage means, per-stage masked
cross-seizure population mean/std, and an identical bounded residual NEZ head.
No electrode/global sample classification or augmentation is copied.

Parity testing loads the pinned official module privately, copies its projection
weights, and compares fully valid sequence outputs. Only compact parity findings
are published. The local upstream checkout is clean and pinned; layer source
SHA256 `268d8d5486685bdc45a1f8c44056a85d7e327f70e973a8b9e50d63675e352c2d`.

Timing source: original feature records carry true relative centers, duration,
annotation onset, valid-start and valid-length metadata. The three shortened
records have51/40/56 windows rather than59. A recording-boundary start is not
necessarily the nominal padded crop origin. Use the explicit relative centers
and valid range, not a forced59-step grid or center-index matching. Clinical
annotation coordinates are source-confirmed, not independently EDF-adjudicated.

All source/cache checks precede real temporal training. The all-cohort source
feature cache includes clinical fields; they are not used in temporal features,
and development is sealed to FIT/VAL before training. Loading a source payload
for metadata audit is not a claim that historical outer labels have never been
materialized. No new outer prediction or performance calculation is permitted.
