# Engineering-only execution record

## 2026-10-17, before end-to-end training

- Original stage1 extraction completed all1,020 patient/encoder cells and wrote
  `EXTRACTION_ADMISSION.json: PASS`. The original PowerShell wrapper read a null
  process ExitCode and incorrectly stopped before screening. The old log is
  retained. No scientific computation was discarded or repeated.
- New `run_checked.ps1` pins the process handle, preserves nonzero-exit failures,
  and requires a separate scientific completion gate even when ExitCode is null.
  It holds an exclusive coordinator file lock. Screening reuses the unchanged,
  fully hash-verified feature seal.
- A separately launched engine unit test aborted during NumPy `blas_fpe_check`
  while importing Torch, before test execution or model updates. The active
  screening process was not affected. `run_tests_checked.py` uses the previously
  established CPU16 affinity and NumPy-before-Torch import order. No package,
  driver, numerical gate, scientific source, input or hyperparameter is changed.
  The replacement test entry must pass before end-to-end admission/training.

These are execution observations, not model performance evidence. They do not
justify a new seed, relaxed tolerance, repeated scientific test or Outcome tuning.

## Completed execution

- The checked entry passed all7 unit tests. Four real complete-patient gradient
  admissions passed: logits/loss exact, maximum gradient error2.68221e-7;
  no formal epoch was discarded.
- Training completed20 cells/224 epochs without trainer errors. One local SSH
  stderr-reader thread timed out after a long empty stderr stream; the owning
  SSH command, remote worker and stdout liveness remained active. No worker was
  restarted because of that observation-channel timeout.
- The frozen evaluation completed320 private patient-score cells and20 fold-model
  cells before reading the full label archive. An independent count-matrix
  bootstrap reproduced all81 absolute and54 paired intervals (10,000 draws each),
  maximum interval error4.44089e-16; all720 patient-metric rows were also checked.
- Local Markdown generation initially lacked the optional `tabulate` dependency.
  The report formatter was replaced by a dependency-free Markdown serializer;
  this changed no predictions, model choices, metrics or numerical artifacts.
- Figure source preflight initially could not detect dynamically named export
  suffixes. Explicit SVG/PDF/PNG/TIFF export calls resolved the source audit.
  A target legend overlapping the bottom data row was moved beside its target
  line during visual QA. No chart observations were removed or changed.
