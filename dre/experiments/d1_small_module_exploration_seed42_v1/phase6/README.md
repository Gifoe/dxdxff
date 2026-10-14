# Conditional feature-complementarity tree residual — Phase6

Earlier completed/verified phases show only0.668015 from patient-context decision shifts and0.667791 from joint readout/cap fine-tuning. They do not reach0.70. This prospective phase tests two fixed nonlinear residual configurations, not another global threshold sweep.

The frozen M8 initializes probability from its existing logit. New boosting trees can interpret processed physiological/context features nonlinearly. Inputs are33 patient-observable M8/D1/context coordinates and the original own-fold88D processed feature bank. Its preprocessing was fitted only on originalFIT patients; no new scaler/data extraction is performed. Exact constant/duplicate pruning reads onlyFIT values and retains explicit feature names. This gives feature-complementarity a genuine supervised test; improvement is not asserted in advance.

Two fixed depth/leaf configurations and all ten original fold cells are required.200 trees per cell are fitted; selection among predeclared0/20/.../200 stages uses only validationMacroF1/EZF1 and the original numeric threshold grid. Original epoch0 frozen M8 is eligible. All tested stages and unfavorable cells are retained, and one global configuration is chosen across folds. Patient-equal channel weights precede the originalFIT class weighting. No identity/center covariate, test label, test threshold or Outer output is used.

Run `code/test_tree.py`, `code/run.py`, then read-only `tools/verify.py` under the same trusted runtime and NumPy-first imports/thread limits. Only after global model selection and fresh checkpoint/preprocessor/threshold freeze may Outer be run. Development has already been viewed/adaptively searched; neither these intervals nor a future repeatedOuter is independent clinical confirmation.

Only source/hash/anonymous reports/results are public. Private patient/channel banks, scores, labels, weights and Raw remain serverD. A local synthetic-test import initially referenced unavailable server-only modules; the import was deferred to the actual training entrypoint before server input freeze. No science parameter or data was changed.
