# Seven unchanged physiological features + Raw temporal dispersion

Completed bounded experiment after explicit user authorization to modify
the Raw architecture on2026-10-17. All10new fits/117epochs and independent
verification passed,but neither new arm passed the internal improvement
gate. No new Outer was run. Actual results are in RUN_REPORT.md and
results/internal_v1;the broader>0.64full80 target remains unmet.
The previous unexecuted patient-context source is preserved, not called run.

The target is full80 **equal-patient Macro-F1 strictly>0.64**, improving the
previous C2 single-head model. That model's actual value0.6396579900303192
does not pass through rounding. Historical Outer has repeatedly been viewed;
future evaluation is exploratory repeated-test, not independent confirmation.

## Mechanism and matched comparison

Both new models preserve the original learned within-window response mean.
A zero-initial24x24 projection adds a bounded nonlinear correction from:

- R0: the same24D temporal mean, a matched extra-capacity control.
- R1: the24D population SD over learned temporal responses.

The correction is before the original pre-ictal latent subtraction, original
7D physiology gate and all subsequent aggregation, not a logit calibrator.
It can alter patient-internal ranking. Both have10,978parameters versus
C2's10,378; the same one110→64→1 scalar classifier receives the96D Raw summary
and original14coordinates. They share all original initial tensors and exact
initial predictions, but their new projection gradients need not match.
Both are trained from scratch; no trained D1 or old C2 initializes them.

The hypothesis is that a learned response's variability within each2s window
can retain transient morphology that its signed average discards. This is
not proof of added clinical information or a novel pooling algorithm. Earlier
Raw-only energy/attention/long-filter negative studies are retained. R1
preserves the mean instead of replacing it with energy, and the present
question is conditional increment with the unchanged7D and C2 head.

All seven original physiological measurements remain, including variance and
partial high-gamma; no new band or clinical descriptor is extracted. Original
Raw250 masks/times/500samples, per-window normalization, same-channel reference,
0–8/8–20 phases, seizure mean/popSD, full-patient96D Z, original Gate and
FIT-only14D preprocessing remain unchanged. Complete gradients and all valid
windows are preserved. Learned within-window SD is not the existing
cross-seizure SD and does not constitute another external feature input.

## Execution and selection

Five exact original C2 checkpoint/validation controls must be hash-verified
and actually replayed. Ten new actual fits use the original seed42, five
patient-held-out roles,30epoch maximum/6minimum/6patience, AdamW and original
Macro-oriented objective. Epoch0 is ineligible. Each complete FIT patient is
one update. No hyperparameter or seed search.

Only original Development chooses epoch/threshold/architecture. Each new arm
must improve its patient Macro by≥.003 versus C2, with AUROC/AP no worse
thanC2−.01. Among eligible arms choose Macro−.1fold sampleSD, then fixed order.
If none qualify, there is no new Outer. A passing candidate still needs a
separate final freeze,single full80 evaluation,independent verification and
actual strict target attainment; internal success alone is not goal success.

Development has65appearances of47people. These same validation labels choose
all decisions, so its results and patient-cluster CIs are selection-conditioned.
Nine metrics,all54absolute/paired intervals,fold/source/coverage tables and
all negatives are reported. Nominal10000seed42 patient-cluster intervals do
not account for adaptation,seed uncertainty or multiplicity.

Trusted-server launch after source deployment and release:

    powershell -NoProfile -ExecutionPolicy Bypass -File D:\m8_7d_raw_dispersion_seed42_v1\run_internal.ps1 -RunTag 01

Keep the foreground SSH owner alive. Resume only after engineering diagnosis,
with a fresh RunTag; never duplicate active work or alter frozen sources.
CPU tests are implementation admission,not performance evidence. Actual
largest complete legal FIT checks precede formal training,then every retained
checkpoint and all patient statistics are independently replayed.

Only source,hashes and anonymous actual aggregates/reports may be published.
Raw data,IDs,individual predictions,weights,scalers,caches and logs stay
on the trusted server. Unrelated jobs and all paused/prior studies are preserved.
