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

The affected host can terminate a Python supervisor together with its GPU child
without a usable Python exit record. The active launcher is therefore a
PowerShell watchdog, which remains outside the Python process tree and applies
the same bounded progress and terminal-failure rules. This changes only host
recovery; the supervised development command remains byte-for-byte the same.

## 2026-09-30 — exact batched rank and order-statistic pooling

The first resumable epoch exposed a second, larger implementation bottleneck:
average-tie ranks were formed by 59 x 68 Python loops per record, including a
CUDA-to-host boolean synchronization for every candidate tie. Record pooling
also invoked a separate quantile reduction for each channel. On the server the
75,473-parameter model consequently used only about 7% GPU compute and required
about 72 seconds per fit patient.

The rank calculation now stable-sorts all window/feature columns together and
derives the same average tie-group start/end ranks with batched cumulative
operations. Masked quantiles now use one stable sort and the exact default
linear interpolation indices for Q25/Q50/Q75, and all channels are pooled over
records in one batch. The definition, valid masks, tie averaging, and pooling
statistics are unchanged.

The server-side CUDA audit used three representative 48-channel x 59-window
records. The former forward/backward path took 41.8818 s and the replacement
took 0.02805 s (1,493x in this isolated synchronization-heavy benchmark).
Maximum logit difference was 9.43e-7, maximum parameter-gradient difference was
1.79e-7, and the rank difference was one float32 ULP (1.20e-7); all are below
the predeclared 1e-6 numerical-equivalence tolerance. Local irregular-mask,
tie, singleton, output, and gradient tests pass. The saved model, optimizer,
RNG, shuffled patient order, and next-patient cursor remain the resume source;
no completed optimizer step is discarded and no validation/test outcome is
used by this engineering change.

## 2026-09-30 — validation metric return-contract repair

After the accelerated path completed the first training epoch, it reached the
validation call for the first time and exposed a dormant runner bug. The shared
validated `ictal_validation` and `omni_validation` functions return
`(aggregate_metrics, private_rows)`, while the PRiSM runner passed that tuple to
a selector expecting the aggregate dictionary. The runner now explicitly
unpacks the aggregate half and continues to retain its existing record-level
rows for validation-only threshold selection. This changes no metric formula,
prediction, selection criterion, model state, or data membership. Epoch 1's
completed patient optimizer steps and their atomic checkpoint are reused;
validation is simply replayed from that frozen state, with no test access.

## 2026-09-30 — VLOO private-row schema adapter

The validation-only VLOO finalizer initially expected each epoch's private
prediction payload to already contain one labels/scores vector per patient.
The trainer intentionally stores the earlier record-level rows instead. The
finalizer now converts those rows using the exact existing Ictal evaluation
rule: group by channel, verify label consistency, average record probabilities,
and sort channel names deterministically. A unit test covers repeated records
and reordered channels. This is a schema adapter only; VLOO epoch/threshold
selection, fixed queries, metrics, memberships, and labels are unchanged. The
failed attempt stopped before writing a public metric file and never opened an
outer/test source.

The historical fixed-query evaluator permits a random half-channel query to
contain only one class. In that case its ranking metrics and balanced accuracy
are undefined and are stored as NaN, while Macro-F1/EZ-F1 remain defined;
aggregate metrics use `nanmean`. The PRiSM finalizer now exactly reuses this
established behavior instead of rejecting such a query. It also restores the
frozen source channel order from each validation patient's already-built token
cache before applying deterministic query membership. This is necessary
because query membership is order-sensitive; alphabetical order would not be
the historical protocol. Token-file hashes are added to the score freeze.
