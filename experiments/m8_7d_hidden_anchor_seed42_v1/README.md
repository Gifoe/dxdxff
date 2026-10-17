# Original7D hidden-anchor single head

Prospective separate trial; the completed balanced experiment remains frozen.
The target is still selected full80 patient-equal Macro-F1>0.64 with the
original7D unchanged. There are no new classification results at registration.

The same final classifier maps normalized14D physiology and96D Raw into one
shared96D hidden layer, then GELU/dropout and ONE scalar readout. Raw enters
as a bounded hidden displacement, not a separately classified logit. The
physiological readout geometry is the exact original duplicate-aware LN88
algebra on14values, not new measurements or naive LayerNorm14.

Three same-capacity variants: scratch joint training; the same head trained
physiology-first then fully jointly adapted; and the same warm head with its
physiological projection/readout fixed while Raw adapts. All preserve the
original Raw CNN/Gate/reference/phase/seizure/PatientZ path and seven
measurements including Variance. No independent trained D1 predictor or D1
output is present. New models never initialize from trained D1/C2 weights.

Five warm fits plus15joint fits are real new training cells. Five exact C2
controls are reused after actual replay, not counted as new fits. The warm
models' feature-only validation is reported as a diagnostic, not a candidate.
Architecture capacity changes versus C2 and extra cheap warm-training budget
are explicitly disclosed; this is not a pure same-budget causal comparison.

All selection uses original FIT/Development, before a separately frozen test
if the unchanged eligibility gate passes. Development is65appearances of47
people and is selection-conditioned. Historical Outer is exploratory
repeated-test, never independent confirmation. All9metrics, all135complete
Development intervals, fold/source stability and real cost/counts are retained.
No target can be claimed from rounding, internal validation or a best secondary.

No arbitrary threshold or architecture tweaks are made to the frozen C2 to
cross its0.000342target gap. This trial tests the physiology-anchor hypothesis
identified before C2's Outer report. It is generic staged training/hidden
fusion, not a claimed novel method or a guaranteed improvement.

## Completed result

All5warm+15joint fits actually completed123+156=279epochs. Independent
verification passed25checkpoint cells,2,925patient metrics and135intervals,
with zero replay/interval difference. Development Macro-F1 isC2.673076,
P.646064,H0.670936,H1.673146,H2.674237. No new candidate met the locked
Macro+0.003gate versus C2; therefore **no new Outer was run**.

See `RUN_REPORT.md`, `COMPLETION_AUDIT.json` and complete anonymous actual
artifacts under `results/internal_v1/`. The overall full80>0.64goal remains
unmet; the old full80C2value0.639658 is unchanged and is not a new result here.
