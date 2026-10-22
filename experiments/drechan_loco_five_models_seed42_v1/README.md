# Four-source LOCO: five fixed models, seed42

Completed and independently verified2026-10-23. All20 main selections are fresh;
16 neural cells ran201 actual epochs in total, with4 final SVMs plus12 OOF SVM
submodels. All20 validation score replays were bit-exact. All20 test cells sealed
before reporting. Verification passed400 patient rows/3600 endpoints,405 nominal
intervals and8 additional-family intervals, tolerance1e-12.

| Fixed model | Patient Macro-F1 | AUROC | AP | EZ-F1 | MRR | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Raw+Feature |0.627380|0.752589|0.557766|0.436358|0.782990|0.675000|
| Raw-only |0.597093|0.710279|0.525347|0.378348|0.781137|0.700000|
| Feature-only |0.613643|0.751334|0.543605|0.415435|0.734570|0.612500|
| Linear SVM, Feature+PR |0.607951|0.738479|0.537266|0.409190|0.728211|0.637500|
| Original TimeConvCNN, Raw/noPR |0.416030|0.574502|0.370946|0.176566|0.468280|0.350000|

All9 endpoints, all20 held-source results and intervals are in public/; see
[FINDINGS.md](FINDINGS.md) for the complete tables and interpretation. Full-minus-
Raw-only Macro-F1/AUROC differences remain positive under the8-test correction.
Full-minus-Feature-only and Full-minus-SVM intervals include zero. Raw-only has a
higher Top-1 point estimate; Full is not best in every source or every endpoint.

Implementation of the user-approved comparison: Linear SVM with six-feature PR,
original raw TimeConvCNN without PR, and current no-guidance DREChannNet using
Raw+Feature, Raw-only, or Feature-only. Variance, both guidance modules and head
interaction remain absent. There are20 fresh main selections, not historical
fivefold checkpoint reuse; SVM additionally fits12 OOF calibration submodels.

## Protocol

Leave out HUP36, LZU21, multicenter15 or pediatric8 patients in turn. Other sources
each reserve ceil(20%) patients for validation using the locked label-free random
assignment. FIT/validation/test counts:34/10/36,46/13/21,50/15/15,56/16/8.
All five models share these roles. Units, imputers, scalers, class weights and
SVM probability calibration are fitted strictly inside the relevant FIT role.
OOF calibration refits even unit-scale estimation inside every subFIT.

Read immutable unscaled R14 and original Raw250 caches in place. Never invoke
old setup/context routines or load old fitted banks, preprocessors or predictions.
Use the original untrained constructor and shared head slices for neural branch
comparisons. Twenty independent validation replays precede the all-model freeze;
all20 test score files must seal before test metrics can be computed.

## Execution

On the trusted server, deploy this directory to D-drive with immutable parent
code present. Python is the existing DREChannNet virtual environment. Run
`run_all.ps1 -RunTag admission01 -StopAfter Admission`, inspect PASS, publish
the admission proof, then run with a new tag and the remote-verified source
commit. Maintain a foreground SSH owner; do not orphan Windows children.
Atomic LAST checkpoints retain optimizer, RNG, complete epoch history and best
selection. Diagnose errors before exact resume; never silently change settings.

`pipeline.py` exposes train, verify_internal, score_test, report, verify, package
stages. `release_check.py` is a local CPU source/package check. Public packaging
uses only the explicit ALLOW list, never a recursive runtime export.

## Interpretation

This is leave-one-*source*-out: multicenter and pediatric are not proven single
hospitals. TimeConvCNN retains its original approximately-29..+29s aggregation,
weighted BCE and BF16 training; DREChannNet retains pre-reference/0..20s stages,
its mixed objective and FP32 training. These are complete pipeline comparisons,
not an architecture-only causal experiment. One seed does not establish seed
robustness. Historical feature/architecture selection used this cohort, so the
results are exploratory and not independent external confirmation. No outcome
direction, improvement or completed result is assumed.

Admission02 PASS:16 matched neural initializers, four complete legal FIT-patient
backward probes and80-patient Raw coverage. TimeConv direct/cache logits/loss
exact, max gradient difference9.313225746154785e-10, BN buffers/RNG exact. Three
CPU tests pass in a Git archive,8 unchanged-parent model tests pass, and separate
synthetic report tests pass. Real results above come from the actual completed
server run, NOT those synthetic tests. See public_admission and RUN_AUDIT.
