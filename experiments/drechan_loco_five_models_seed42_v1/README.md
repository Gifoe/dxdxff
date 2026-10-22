# Four-source LOCO: five fixed models, seed42

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

Status2026-10-22: real-server admission02 PASS,16 matched neural initializers,
four complete legal FIT-patient backward probes and80-patient Raw coverage.
TimeConv direct/cache logits and loss exact, max absolute gradient difference
9.313225746154785e-10, BatchNorm buffers and RNG exact. Three CPU tests also pass
in an actual Git archive. Formal20-cell fitting follows the verified source push;
no test outcomes exist yet. See public_admission and RUN_AUDIT.
