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
