# Early Raw FIR execution record

## 2026-10-17 admission and import-phase interruption

Prospective scientific source was pushed at
`df7e21e690856f46b9e5b24a9c3e49f74e11626f` before admission/fitting.
The eight scientific files on the server matched the committed source hashes.

Actual admission passed all six tests, all seven retained SHORT_MEAN control
checkpoint replays and all four largest-complete-legal-patient backwards.
Each actual control replay had maximum logit error zero and all 201 CAL
threshold decisions were identical. The largest patient had 164 channels and
48,380 valid windows. All four models had finite, nonzero backbone/head
gradients; both extended filters had nonzero gradients. Matching-pool initial
logits and shared gradients were exact on the GPU (maximum difference zero).
Admission performed zero optimizer updates. These checks are correctness and
admission evidence, not classification-performance evidence.

Public admission SHA256:
`4ae63018fff58a20c14ffea33fd382886244baf00fe56d75333144626cb790f8`.
Private input-freeze whole-file SHA256 (contents stay private):
`ecf6ee0475aa28f59ce3a5b90b8514670b6e6ede1aa3e62a6ada15cf85901bfb`.

The first formal worker stopped while importing PyTorch's
`torch.fx.experimental.proxy_tensor`, with
`SystemError: error return without exception set`. It did not reach setup,
create a model directory, make an optimizer update or read new assessment
outcomes. The coordinator returned nonzero and did not auto-retry. Logs were
preserved. No matching recent Windows Application/System error was returned
by the targeted event queries. An isolated fresh-process import of the exact
runtime, PyTorch and task common module passed, without setup or training.
The underlying cause is **not proven**; do not relabel it a CUDA/model error.

The permitted recovery is the identical frozen coordinator with a fresh log
tag, preserving the completed admission and all source/input hashes. No
dependency, math, model, data, role, seed, optimizer, threshold, selection or
budget is changed. A repeated import failure must be diagnosed rather than
handled by an unattended retry loop.
