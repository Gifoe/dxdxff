# Actual execution audit

Executed on 2026-10-17, Asia/Shanghai, after explicit user authorization to evaluate all existing candidates without further Development screening.

## Fixed artifacts

- Source/protocol commit pushed before Outer inference: `59a1f770698e94bae9a880b713535d587d613d5d`.
- Deployment archive SHA256: `e7c50a54ffcc444bd168d3fe86a6244b59f8a7145836ab6e4132868a34d04fb3`.
- Pre-Outer private freeze SHA256: `f8205e35ca983e49c69265e5c07921d0b9e10eff46c7b89d524bb41f43a81388`.
- Complete 55-cell private score seal SHA256: `b345ac3a2aa934c97782280275a149737d96d870b5e54eadd0b629f77ef44bf6`.
- Public allowlisted export SHA256: `bd3028f11f26c4b0551e4ad0835cb2796024e183d2ff62b6c3b3800704527004` (34,019 bytes; 16 public artifacts plus manifest).
- Original C2 score seal remains `9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c`.

Historical model/candidate/input/source hashes, the 55 selected checkpoint hashes, epochs and thresholds are in `results/ADMISSION.json` and `results/FROZEN_MODELS.csv`. Earlier sources, checkpoints and failed Development gates were not overwritten.

The aggregate export manifest records the exact server bytes. The results directory disables Git line-ending conversion so that every published blob retains those original SHA256 values; frozen executable source uses LF. This is packaging only and does not change any prediction, numerical result or model binding.

## Actual sequence

1. CPU AST / PowerShell parse checks passed for the new audit code. Server tests for the fixed arm contract, exact private cache identity/tamper rejection, and paired NumPy bootstrap ordering: **3 tests PASS**.
2. Read-only replay of the first existing validation patient for each of 55 checkpoints: **55 PASS**, maximum logit error **0**, all stored-threshold decisions identical. No labels were used by this replay, no performance screen was run and no selector was invoked.
3. Admission and all model/threshold/preprocessor/source bindings frozen. Source committed and remote GitHub tip verified before new Outer inference.
4. Sustained foreground server coordinator ran `run_outer.ps1 -Stage All -RunTag 02`. It rejected duplicate workers and maintained a private runtime lock. All ten candidate arms completed five frozen inference cells each. Five existing C2 score cells were reused exactly. There were no crashes, retries or incomplete candidates in this Outer execution.
5. All 55 score cells were sealed before the full-cohort metric reporting stage. Every new arm's Raw coverage was verified: 80 patients / 7,635 channels / 1,471,965 valid windows, corresponding to the original 256-seizure cohort.
6. Reporting and independent verification completed; all numerical checks passed. Admission, inference, report and verification worker stderr files were empty. The test runner's stderr contained its normal `3 tests ... OK` output, not an error.
7. Export traversed only an explicit sixteen-file public allowlist. Local archive SHA and each manifest file SHA were verified; the public-only aggregate verifier passed. No private artifacts were fetched.
8. The server had no remaining audit Python process after completion; sampled GPU was idle. Local unexecuted patient-context proposal was preserved and excluded from commits.

## Exact terminal

Coordinator: `ALL11_FROZEN_FULL80_OUTER_PIPELINE_VERIFIED_COMPLETE`

Independent verifier: `ALL11_FULL80_OUTER_INDEPENDENT_VERIFICATION_PASS`

Public verifier: `ALL11_FROZEN_OUTER_PUBLIC_AGGREGATES_PASS`

`OUTER_STATUS.json`: `COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT`.

`INDEPENDENT_OUTER_VERIFICATION.json`: `PASS`, 880 patient rows, 7,920 patient metric checks, 234 nominal interval checks, ten additional Macro-F1 family interval checks, maximum nominal interval difference 6.661338147750939e-16.

The terminal denotes actual execution/verification completion. It does not mean that a candidate passed its old Development gate, that the largest observed Outer result was prospectively selected, or that independent statistical superiority has been demonstrated.
