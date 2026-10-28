# PR²: patient-relative ranking and calibration

Fixed, exploratory Seed42/original fivefold experiment on all80 ictal patients,
7635channels and256seizures. No new extraction or architecture search.

StageA freezes the historical12D linear SVM and compares NO_PR, LegacyPR,
output-logitZ, a capacity-matched expanded-ABS adapter, ranking-only PR² and
complete PR². New adapters use legal3patient-fold nested cross-fitted FIT logits.
StageB jointly trains the specified original six-feature FULL DREChanNet with
ranking loss, expandedABS or PR², with identical initialization and budgets.
The old FULL reference is seed42,11425parameters; it is not the later no-guidance
architecture or the historical three-seed average.

The16explicit scientific files in `runtime.source_hashes()` and
`PROTOCOL_LOCK.json` fix all calculations before formal fitting. The source
commit is pushed first. Final report layout/packaging can be added afterward;
they consume independently verified anonymous tables, never select models.

Trust-server execution: use the existing pinned connection and
`D:/Codex_Research/tools/Run-Legacy.ps1` to run `run_all.ps1` from the canonical
experiment directory. `Admission`, `StageA`, `StageB`, `All` are supported.
Use a fresh RunTag for a diagnosed engineering resume and preserve model,
optimizer,RNG and all existing logs. A single coordinator lock prevents duplicates.

SVM historical replay permits at most1e-12 arithmetic roundoff only when the
feature hash, every201threshold decision and all patient ranking orders are
exact. Historical row summaries use the original sealed probabilities. DREChanNet
disabled-wrapper and frozen validation-logit parity remain bit exact.

Portable tests: `python -m pytest tests/test_pr2.py -q` or `python tests/test_pr2.py`.
Server admission adds real FIT-only global-reference calculation, historical
validation replay and complete-patient CNN-gradient checks on allfivefolds.

All source results, including unfavorable arms, will be reported. Clinical
identities, raw signals, individual predictions, embeddings, reference arrays,
weights, scalers, optimizers and logs remain private. Only anonymous aggregates
and integrity evidence are eligible for the public release.
