# Engineering change log

## 2026-09-30 — masked quantile pooling vectorization

The initial implementation evaluated temporal quantiles one channel at a time.
That created three CUDA quantile launches per channel per record and made the
first development epoch impractically slow. The pooling definition has not
changed: it is still the mean, Q25, Q50, Q75, and maximum over exactly the
valid windows/records. The implementation now applies NaN-masked vectorized
reductions and requests all three quantiles in one batched operation, which
exclude the same invalid entries.

The executable audit compares the vectorized output to the prior explicit-loop
rule on irregular masks and singleton support, and verifies finite gradients.
No data, feature, architecture, loss, optimizer, selection, threshold, or
split setting changed. Any pre-epoch partial computation is discarded; the
label-independent token cache and train-only scaler are retained.

## 2026-09-30 — patient-step crash recovery

The server's NVIDIA driver terminated the Ictal process with a native
`nvcuda64.dll` breakpoint before its first epoch completed. This was not a
Python/model error and produced no checkpoint or validation result. Training
now atomically records the model, optimizer, RNG states, fixed shuffled patient
order, cursor, and accumulated observed labels after each completed patient
optimizer step. A restart resumes that exact state; it does not replay a
different epoch order, alter an optimizer update, select a checkpoint, or read
test data. Epoch-level selection and early stopping remain unchanged.

The restart supervisor is bounded and only retries the exact development command
when its captured stderr contains the already observed native Windows/NVIDIA
failure signature. Python-level and unknown failures stop for diagnosis rather
than being retried as though they were transient.

## 2026-09-30 — host-level recovery of killed supervisors

On this Windows host, a native GPU interruption can also terminate the Python
supervisor before it has a chance to record the child exit code or terminal
status. A separate standard-library host watchdog now relaunches that exact
supervisor, keeps host-attempt logs separate, and fingerprints the private
patient-step state after every interruption. It stops after three consecutive
interruptions that made no resumable progress, and immediately stops on any
inner Python/non-native failure recorded by the supervisor. It does not import
the model, load data, or alter training, validation selection, or test access.
