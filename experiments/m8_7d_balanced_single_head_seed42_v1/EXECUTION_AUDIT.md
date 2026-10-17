# Original7D single-head execution

## Registration and actual admission, 2026-10-17

Source/protocol was pushed before fitting at
`ab14aa43e5d2cc5bae5f1146e2f91f8e1078a5f1`, branch
`codex/m8-7d-single-head-macro-seed42-v1`. A separate worktree preserves the
paused Raw-only study and its uncommitted results, and unrelated user changes.

Four unique CPU correctness tests actually passed in0.08s. The complete task
module imported before setup without creating a training run. Actual server
admission then passed the same four tests, all five existing S3 validation
checkpoint replays, and three full actual-patient gradient checks. Maximum
replay logit error was zero on every fold and all201 threshold decisions were
identical. Source/input/role/checkpoint hashes were checked before reuse.

The largest complete legal FIT patient had164channels and48,380valid Raw
windows. All three variants reproduced the original Raw summary exactly;
padding and channel-order checks passed. All seven physiological coordinates
remain enabled. CNN/head/Gate gradients and the C2 interaction-gain gradient
were finite and nonzero as required. Gate first-layer gradients were checked
after the original zero-output Gate's first step. Exactly three legal FIT
smoke steps per model were discarded; no formal epoch was discarded.
These correctness checks do not constitute classification-performance results.

Public admission SHA256:
`42c5e0dd07da7a514ab9dc4ddbeda9d9bec08478f8763d027087dd9b4160df2f`.
Input binding digest:
`8d8ff0e27f485454a61acbb022f7c1153e09972bb88f47a73698c1a1a0456803`.

Run tag01 uses a live foreground SSH owner. The coordinator has no automatic
retry and writes fresh separate stage logs. Only two new candidates×five folds
are fitted; exact existing S3 controls are not refitted. All training, inputs,
epoch and threshold selection retain the original protocol. Architecture
selection is Development-only and must precede any new Outer inference.

## Completed Development and prospective final-stage registration

Tag01 ended normally with
`BALANCED_7D_SINGLE_INTERNAL_PIPELINE_COMPLETE_NO_OUTER`.
All10newfits completed114epochs. Five old controls were replayed rather than
retrained. Independent replay of all15models had maximum logit error zero;
all1,755patient-metric values and54cluster intervals passed independent
verification with maximum interval difference zero. There were no training
or verification stderr messages and no new Outer access.

The internally fixed candidate is C2_INTERACTION. C1/C2 both passed the
registered eligibility rule; C2's exact penalized score was higher. Internal
model freeze SHA256:
`c4aaecb2be9dbaa6f0b0da0351a7fa0b3eec13057a763c5b863ed5b613192e10`.
Candidate SHA256:
`51f89e56f67c4d8ce22324c56e1e143e23b7f6673db1224fd93f8b3c1aef4d72`.
The13file anonymous internal delivery ZIP was verified as
`26a8ec6ab6dab9768ee2b0615c78a7df73fe86dc6de0dc4221284547fbf5ad8d`.

Final-stage source compiled locally and its PowerShell launcher parsed.
The exact five new final source files imported in the server runtime without
setup, training or test access. Source ZIP SHA256:
`0f95f4a22427139483959ee3b4b3eecd64f623eb33cfaf73eb07941607cac1c3`.
This import check is not evidence that final inference/statistics succeeded.
Final protocol/source are published before the separately frozen test pass.
