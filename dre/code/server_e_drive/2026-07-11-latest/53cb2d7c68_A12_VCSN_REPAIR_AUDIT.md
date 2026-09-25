# A12-VCSN repair audit

This audit was performed against the in-place A12 implementation before the repair.
The frozen V3 all90 outer-fold ledger remains the source of truth; no result in
this document is a new experimental result.

| current_file | current_behavior | defect | planned_fix | test_to_add |
|---|---|---|---|---|
| `a12_vcsn/suite.py` | Uses only `seeds[0]`, calibrates probabilities but retains raw utility, and selects pair-mean thresholds. | Seed, calibration and policy protocol are invalid. | Add seed aggregation, post-calibration utility recomputation, and patient-level inner-OOF action simulation. | multi-seed, calibration-utility, policy-objective |
| `a12_vcsn/models/trajectory_tcn.py` | Encoder is not an optimised module and utility model inherits Siamese unchanged. | V8/V9 are aliases rather than temporal models. | Replace with trainable masked temporal-convolution pair model. | TCN trainable, gradient, V8-not-V6 |
| `a12_vcsn/models/siamese_utility_mlp.py` | Splits one flat vector at its midpoint. | Eject/add semantics and scaler boundaries are implicit and unsafe. | Use named eject/add/pair/patient feature blocks with block-local scalers. | explicit blocks, direction, holdout scaler |
| `a12_vcsn/models/ensemble_utility.py` | TCN member resolves to the Siamese alias; no inner membership selection. | V10 lacks three distinct members and leaks selection intent. | Use distinct CatBoost/Siamese/TCN members; select combinations from inner OOF only. | distinct members, inner selection |
| `a12_vcsn/io.py` | Concatenates all run trajectories per channel. | Seizure/run boundaries are lost. | Add canonical `PatientTrajectoryStore` and per-run aggregation. | run boundary, masks, padding |
| `a12_vcsn/trajectory_features.py` | Only global mean/std/quantiles. | Required within- and across-seizure structure is discarded. | Add masked compact/full trajectory summaries and ranks. | trajectory full, NaN/Inf |
| `a12_vcsn/audit.py` | Defaults expected parity metric to recomputed anchor metric. | Anchor metric can self-validate. | Require independent expected metric or classify as contract-only. | self-validation refusal, known metric |
| `a12_vcsn/candidate_pool.py` | Uses only selected-tail and V3 boundary sources. | Candidate source union and source accounting are incomplete. | Add label-free source flags, quotas, pair caps and diagnostics. | candidate union, cap/coverage |
| `a12_vcsn/patient_gate.py` | Gate target is the best pair label. | It does not score the matched patient action set. | Build one inner-OOF action-set row per patient and target net executed delta. | action-set target |
| `a12_vcsn/reporting.py` | Mislabels patient summaries as deltas/fold metrics and emits a placeholder report. | Outputs do not meet their stated semantics. | Write actual deltas, fold/center summaries and generated report. | report semantics |
| `a12_vcsn/protocol.py` / `a12_vcsn/suite.py` | Resume hash covers config only; provenance is incomplete. | Changed frozen inputs can resume incorrectly. | Build input/schema/registry/git fingerprint and per-fold provenance. | fingerprint invalidation, no-leakage |
| `scripts/audit_a12_vcsn_inputs.py` / `scripts/run_a12_vcsn_suite.py` | No independent parity CLI parameters; several runtime options are inert. | Audit and execution cannot establish required protocol. | Wire all requested CLI/config fields and write parity artifacts. | CLI parser/config |
| `exp_ez_hybrid.py`, `patient_channel_ranker.py`, `temporal_encoder.py`, `seizure_aggregator.py` | Existing repository temporal/ranking components establish patient/fold conventions. | A12 did not document reuse boundary. | Keep A12 read-only over their frozen outputs; record this boundary in provenance. | provenance manifest |
| `neuroez_c/dataset.py`, `neuroez_c/protocol.py`, V3 export/fixed-all90 modules | Define canonical cache/fold and fixed-all90 protocol constraints. | A12 only partially enforced those constraints. | Validate canonical joins and source fold assignments before fitting. | fixed-fold and coverage checks |

## Repair acceptance boundary

The changes below are implementation and synthetic-regression work only. The
remote all90 cache is not present in this workspace, so no claim about a real
five-fold score, including a score above 0.70, is valid until the prescribed
remote audit and full suite are run.
