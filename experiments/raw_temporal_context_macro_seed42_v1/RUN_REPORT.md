# Raw temporal-context adapter — actual internal results

## Outcome and scope

**No fold passed the registered advancement gate.** All 15 first-path neural
fine-tunes finished, totaling 126 epochs. Independent selected-state replay
passed with maximum logit difference zero and all nine patient metrics checked.
No original Development fit or new Outer evaluation was run. The broader
goal, complete 80-patient equal-weight Macro-F1 >0.65, remains unmet.

This table concerns legal internal patient-held-out assessments: **86 patient
appearances / 59 unique people per arm**, not the complete 80-patient Outer.
It must not be compared directly to the historical Outer reference 0.630493
as though both numbers described the same evaluation population. The original
Raw banks, 80-patient/7,635-channel/256-seizure/1,471,965-window contract and
fivefold role identity are unchanged; no quality-based exclusions were made.

## Matched comparison

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unchanged Native | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 |
| Frozen precompand body | 0.604982 | 0.723093 | 0.535747 | 0.382907 | 0.827057 | 0.745545 | 0.746988 | 0.627907 |
| Same precompand body fine-tuned | 0.605607 | 0.724792 | 0.537415 | 0.383582 | 0.827632 | 0.746132 | 0.754256 | 0.639535 |
| Equal-size channel MLP | 0.605650 | 0.724769 | 0.537375 | 0.383582 | 0.827718 | 0.746258 | 0.754256 | 0.639535 |
| Temporal-context residual | 0.605692 | 0.724834 | 0.537396 | 0.383647 | 0.827736 | 0.746234 | 0.753979 | 0.639535 |

All values come from actual completed fits or the exact unchanged baseline
scores; no unrun second/third path or Outer result is filled in. Full precision,
EZ-fraction bias, fold means/SD, source aggregates and paired intervals are in
`results/`. Source outcomes with fewer than five unique patients are suppressed.

Paired 10,000-patient-ID-cluster bootstrap (carry repeated appearances together):

| Comparison | Macro-F1 difference | Nominal 95% interval |
|---|---:|---|
| Temporal minus Native | +0.003718 | [-0.005222, 0.012789] |
| Temporal minus frozen precompand | +0.000709 | [-0.000481, 0.001904] |
| Temporal minus matched fine-tune | +0.000085 | [0, 0.000266] |
| Temporal minus equal-size MLP | +0.000042 | [-0.000125, 0.000254] |

Temporal-minus-MLP AUROC is +0.000066, interval [-0.000043, 0.000219].
The study does not establish a meaningful temporal-specific benefit. Most of
the numerical gain over Native was already present in the frozen two-waveform
body, not supplied by the new temporal block. The interval with a zero lower
bound must not be described as evidence for a substantial improvement.

## What changed and what did not

The model remains a shared Raw CNN with one classification head. Both existing
waveform normalizations feed the same encoder; the source bounded 24x24 fusion
matrix, pre-latent reference subtraction, two temporal phase means, available-
seizure mean/SD and complete-patient Z are retained. No handcrafted bandpower,
variance, entropy, physiological descriptor, D1 score or identity predictor is
added. The pre-RMS scale is used only inside the already fixed waveform
normalization, not appended to the classifier.

The new residual acts before window global averaging. The pointwise channel
MLP and temporal depthwise/pointwise block each add exactly 984 parameters.
The first has 24→20→24 channel mixing; the second has depthwise kernel17
followed by 24→24 pointwise mixing. Both output layers start at zero, giving
the same initial logits as the precompand body. The temporal local receptive
field grows from 33 to 289 samples. Both models already globally aggregate
all 500 waveform samples; no new whole-window coverage is claimed. Generic
temporal residuals are not claimed novel.

The original two-waveform body has 13,105 parameters; both adapted models
have 14,089. The original Native has 12,529. All three trained arms thaw the
same CNN/head/source matrix with the same registered optimizer, Macro-weighted
loss, patient order, CAL threshold grid and epoch budget. Every patient update
uses all valid windows/channels/seizures, including full pooling and Z gradients.

## Selection and negative result

All three arms selected epoch2 in fold1 and epoch0 in folds2–5. Thus 12/15
selected models kept the unchanged warm initial state, although all 15 fits
actually executed their registered epochs. No losing epoch was forced into
the result to make a module look different.

The first-path fold gate required a candidate Macro gain of at least 0.01 over
both unchanged baselines, plus the AUROC safety condition. Maximum observed
fold Macro gain was only +0.002396 in fold1; folds2–5 did not improve the
stronger frozen baseline. All five choices were therefore frozen to Native.
Per the prospective protocol, no extra paths or new Outer were justified.
The unexecuted final-stage plan is not presented as a completed result.

The warm initial CNN and fusion matrix had already been CAL-selected on the
same path. Retention of epoch0 therefore limits interpretation: this is a
negative result for this warm adaptation/selection protocol, not a proof that
longer temporal morphology is uninformative or that all Raw-only models cannot
exceed 0.65. A future from-scratch Macro-aligned representation experiment
would be a different protocol, not a budget extension or rescue of this run.
The earlier 535-sample receptive-field experiment used EZ-first optimization
and also did not establish Macro improvement; it is explicitly retained in
[the mechanism evidence ledger](MECHANISM_EVIDENCE.md).

## Admission, verification and compute

Four changed-critical-path source tests passed. One pytest warning concerned
conversion of a test-only autograd tensor to a scalar; it did not mutate the
model, skip a test or alter formal data. The original registered test source
is preserved. No failed numeric gate was widened.

The largest complete patient among 79 actual training identities had 164
channels and 48,380 valid windows. All three models completed direct forward/
backward with finite gradients, exact initial logit identity and shared-gradient
agreement at the fixed tolerance. Peak allocated GPU memory was 1,777,326,080
bytes for the fine-tune, 2,832,324,096 for the MLP and 2,947,763,200 for temporal.
These cold/warm admission timings are not a fair speed benchmark.

All 15 initial CAL replays had zero logit/probability error and unchanged
decisions across all 201 threshold candidates. Formal epoch wall time averaged
2.374677 seconds, totaling 299.209275 seconds across 126 epochs (excluding
preparation, HELD inference, replay and other orchestration). No unrelated GPU
job was stopped or altered. No engineering/native restart was needed.

The independent verifier checked all 15 selected model states, exact source
body/matrix/role hashes, CAL re-selection, HELD predictions, all nine sklearn/
ranking patient metrics, all race decisions and 430 private patient metric
rows across 25 logical cells. Maximum replay error was exactly zero. It
confirmed that original Development and Outer were excluded. Patient identity
and raw details remain private; the public verifier checks anonymous aggregate
consistency and byte hashes, not a substitute for private Raw replay.

## Provenance and delivery

- Internal score seal SHA256: `2c1edbac1dbc1c85e44cb3b7fb330f25545e4f1a1a7729fcae58c5a3ce5d808b`
- OwnFIT selection SHA256: `4927bca76c9f857bf8e93438330404f12deb67e417587f82078f23405fc0e967`
- Independent terminal: `COMPLETE_VERIFIED_TEMPORAL_CONTEXT_INTERNAL` / `PASS`.
- New original Development fits: 0. New Outer passes: 0. Target achieved: false.

Earlier Outer results remain exploratory repeated-test. Internal paths are also
historically reused and overlapping; nominal cluster intervals do not create
independent confirmation. All old sources, freezes, checkpoints, predictions
and logs are preserved. Public delivery contains only code, hashes, anonymous
statistics, this report and the cross-study evidence ledger. Private identities,
labels, raw banks, individual scores, weights and logs stay on the trusted server.

Server reproduction: `run_internal.ps1 -RunTag <fresh>` in a sustained
foreground SSH owner. Public delivery check: `python verify_public.py`.
