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
