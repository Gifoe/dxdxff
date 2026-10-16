# Internal information-retention execution ledger

## Before formal training

- Initial source/recipe registration: `a48b2eb`. Four unit tests passed locally
  and on the trusted server. The label-blind own-FIT audit covered79 unique
  patients and1,455,327 genuine Raw windows; no channel/window was removed.
- Original nondeterministic direct/cache admission failed on long-CNN
  `cnn.2.weight`, reported maximum difference1.10361725e-6 under the fixed
  atol1e-6/rtol1e-5 criterion. No formal training epoch had started.
- Diagnosis observed direct autograd chunk order11 through0, versus the cache's
  ascending order. Nondeterministic direct self-replay also varied. Deterministic
  cuDNN plus reverse replay gave direct self/cache gradient error0 in both
  tested long-CNN variants. The additional cross-mode comparison FAILED for C2
  (2.59280205e-6 gradient difference despite exact forward outputs). It remains
  a FAIL, not a claimed equivalent backend replacement.
- A streamed multiline PowerShell deployment command did not apply the source
  cutover. A subsequent admission-only command therefore still used v1; that
  run completed no formal training and was not accepted as the v2 admission.
  Source inspection detected the mismatch. The saved `deploy_execution_v2.ps1`
  verified archived/current/incoming hashes and applied the actual cutover.
- Revision `daa6d9c` prospectively fixed deterministic-cuDNN execution for every
  arm before fitting, retained the same full networks/data/loss/threshold rules,
  retained original tolerances, and preserved v1 source/audit/logs. New outputs
  use the separate `deterministic_execution_v2` subdirectory. No package or
  driver was changed; cuDNN was not disabled. No old formal epoch or weight was
  discarded or resumed under an incompatible backend.
- The actual v2 audit rechecked the exact source hashes. All9 actual complete-
  patient admissions then passed with logit/loss/all-gradient error0 and exact
  RNG parity. Formal epochs discarded remained0. This validates direct versus
  cached computation within v2, not cross-backend numerical equivalence to v1.

No admission/diagnostic values are classification evidence. Neither original
Development nor any new Outer outcome was used to choose this execution repair.
