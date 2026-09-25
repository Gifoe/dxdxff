# A12-VCSN Final Implementation Audit

| requirement | current implementation | defect addressed | fix | verification | status |
|---|---|---|---|---|---|
| Runtime-only paths | `scripts/run_a12_vcsn_suite.py` | Paths previously required only as CLI values and the required policy alias was absent | CLI, JSON/YAML, and `A12_*` environment resolution; placeholder-only example config; both local/server Windows paths accepted | `test_a12_runtime_paths.py` | implemented |
| Cache filtering and privacy audit | `cache_filtering.py`, inspector and fixture scripts | Duplicate groups were not fully dropped and shape/dimension/finite checks were incomplete | Read-only full-group filtering, hashed row audits, five required output files, source hash before/after | cache filtering tests plus real-cache audit | implemented |
| Frozen Task-1 protocol | `schemas.py`, `protocol.py`, `evaluation.py` | None of the final repairs may change source folds, labels, or frozen selections | Source `fold_idx`, explicit label/prediction semantics, authoritative patient evaluator, strict parity | protocol and anchor parity tests | implemented |
| Normalized channel key | schemas, candidates, pair features, HNC, matching, action application | Several joins/actions used original strings | All internal joins/actions use normalized names; public original and normalized names retained; prime contacts preserved | channel normalization tests | implemented |
| Anchor fail-closed coverage | `anchor_features.py`, suite audit artifacts | Anchor returned no coverage table and silently tolerated missing/nonfinite features | Tuple API, per-patient hashed coverage, strict failure/non-strict unavailable state, no label use | anchor coverage tests | implemented |
| Multi-feature relative trajectory | trajectory store/features and suite | Capability was inferred from `trajectory_mean_f0` and lacked patient/run relative ranks | 20-feature capable patient/run ranks, top-q, stability, leave-one-seizure-out, masks, feature mapping | trajectory relative-rank tests | implemented |
| Seizure-aware TCN | `models/trajectory_tcn.py` | Full-batch/static fallback and padded normalization violated the hierarchy | `[B,S,T,F]`, only B*S structural folding, trimmed masked convolution, cross-seizure pooling, no static success | TCN hierarchy and model tests | implemented |
| Complete model config/device | config, model factory, all members | Fixed/partially propagated training values and ineffective device setting | Full config propagation, strict CUDA policy, tensor/model device movement, audited manifests | training pipeline tests | implemented |
| Mini-batch/validation/early stop | Siamese and TCN models | Fixed-epoch full-batch fitting and scaler leakage | Patient-grouped split, balanced sampler, train-only scalers, mini-batches, best restore, checkpoints/logs/manifests | training pipeline tests | implemented |
| Learned patient gate | `patient_gate.py`, suite | Per-action fixed-threshold gate and incorrect counts | One-row action-set summaries, StandardScaler + logistic cross-fit, train-only threshold search, all-or-none test action sets | gate cross-fit tests | implemented |
| Topology candidates | topology and candidate modules | Missing isolated/segment/same-shaft source generation | Valid-topology-only sources, independent flags, deterministic cap after union | topology candidate tests | implemented |
| HNC semantics | HNC loader and CLI join | Exact original-name join and inferred direction | normalized fold-aware join; explicit p/residual EZ/NEZ semantics; strict fail or V5-only non-strict skip | HNC semantics tests | implemented |
| Full policy grid | `thresholds.py`, action simulation | Fixed swap range and conflated harm metrics | Registered formal grid, `0..max_swaps`, complete matching/evaluation path, distinct patient/action harm and coverage metrics; explicit smaller smoke grid | policy tests and all-variant synthetic run | implemented |
| CatBoost inner selection | CatBoost model and suite | No registered grouped inner selection or early stopping | Exact depth/rate/L2 grid in formal mode, grouped early stop, inner-OOF policy objective, selected-param artifacts | synthetic integration and selected-param artifacts | implemented |
| V10 selection/uncertainty | ensemble and suite | Member-selection uncertainty incomplete | Seven combinations, inner-only policy selection, selected members on test, seed/member/total uncertainty | all-variant synthetic run | implemented |
| No-leakage provenance | protocol, calibrator, model/gate manifests | Generic train-subject placeholders | Actual role-specific fit/selection subjects and fold-scoped no-leakage JSON | provenance tests | implemented |
| Resume fingerprint | provenance, suite, pair cache | Config-only resume check and environment snapshot masquerading as Git state | Input/filter/schema/HNC/summary/registry/config/Git hashes, differing-field refusal, variant completion fingerprint, real force rebuild | resume fingerprint tests | implemented |
| Standard reporting | reporting module | Noncanonical patient deltas and incomplete bootstrap/action summaries | Required patient schema, fold/center/action/delta matrix, CI and positive probability, explicit run classification | reporting schema tests | implemented |
| Synthetic V0-D0-V10 | deterministic synthetic capability builder | V5 and trajectory variants previously skipped or silently degraded | Multi-seizure 20-feature store, masks, HNC, anchors, all registered variants executed | `test_a12_all_variants_synthetic.py` | implemented |
| Real-cache fixture CPU smoke | inspector, fixture builder, suite CLI | Earlier TCN masked-seizure gradients could become nonfinite | Bounded anonymized fixture, finite masked attention/variance, explicit CPU smoke mode | V2/V3/V4/V6/V7/V8/V9/V10 strict CPU smoke | verified |

Formal full five-fold results require the external frozen old-V3 ledger and an independent old-V3 metric summary. Synthetic, fixture, and partial executions are explicitly classified and cannot support a clinical performance claim.

## Final verification evidence

- `python -m compileall -q a12_vcsn scripts tests`: exit 0.
- `python -m unittest discover -s tests -p "test_a12_*.py" -q`: 79 tests passed.
- `python -m pytest -q tests -k a12`: 110 passed, 422 deselected, 7 subtests passed.
- Real cache audit: 501 input runs, 155 patients, 20 features; 29 channel-axis/source exclusions plus both rows of one duplicate group; 470 retained runs, 149 retained patients, 6 zero-valid-run patients; source hash unchanged.
- Strict real-structure fixture CPU smoke: V2, V3, V4, V6, V7, V8, V9, and V10 all succeeded.
- Formal server cache, HNC ledger, and pipeline paths were absent on this machine; real one-fold and full five-fold training were therefore not executed.
