# Active goal: M8 physiology feature discovery, seed42

The full requested objective remains active. Data extraction or linear screening is not completion. A null result must be reported honestly within the tested candidate class; finite experiments cannot prove that the original7D is globally optimal.

Original objective: 以M8为基准，自主审计现有7D生理特征的必要性、冗余性与跨患者稳定性，结合最新研究探索更具SOZ定位价值的新特征。自主完成特征构建、筛选、消融、模型整合及性能验证，严格避免测试集信息泄漏。最终目标是在保持模型轻量化的同时提升AUROC、AP和F1，并通过可复现实验明确哪些特征真正有效、为何有效，以及是否优于原始7D；如果能找到需要依据，找不到需要证明现有的合理性。

## Completed execution in this goal turn

- Revalidated original code/data provenance, patient-held-out partitions and exact seven-feature window registry. No new Outer outcome read.
- Targeted primary-literature review includes2026 multi-band peri-onset localization and2024 complexity/spectral-decomposition studies; separate clinical targets and unsupported signal requirements are identified in `LITERATURE_EVIDENCE.md`.
- Locked11 new cache-supported feature formulas before extraction. Analytic/scale/constant/mask/singleton unit tests passed.
- Ran225 original7 linear subFIT/calibration/screen probes (full,seven single groups,seven deletions) inside originalFIT only; no original validation labels used for feature screening.
- Label-blind extraction actually completed for80patients/7635channels/256seizures/1471965valid channel-windows, finite100%; existing actual7D window deltas reaggregate exactly to R14.
- Ran390 prospective candidate probes inside the same originalFIT screens. Retained private classifiers,scalers,probabilities,labels and indices for independent verification. No global feature winner selected across original validation or Outer.
- Independent verifier replayed all390 candidate probability/threshold pipelines and confirmed all80 candidate caches against original aggregation; max replay and aggregation errors0. Only compact anonymous tables were fetched into `initial_results`.

## Completion requirements still pending

| Requirement | Current evidence | Required remaining proof |
|---|---|---|
| Old7 necessity | Linear screens plus completed16×5 actual D1+M8 validation retraining | Independent replay of all160 neural models; final fixed Outer ablations and uncertainty |
| Redundancy | FIT-only Spearman and numeric-scale tables | Distinguish algebraic duplication,correlation,conditional utility and normalization geometry; verify any pruning by matched retraining |
| Cross-patient/center stability | All705 nonlinear patient screens and520 FIT-center holdouts independently verified,including orientation and uncertainty | Integrate these bounded findings with final neural evidence; no pooled-center shortcut |
| New feature construction | Actual11-candidate complete extraction | Independent source/cache/math checks and end-to-end numerical gates; no LF/HFO claims outside source support |
| Feature selection | Nonlinear/center per-FIT subsets independently reproduced and frozen | Complete fixed combined/pathway/negative-control neural procedures; no global validation-selected feature winner |
| Lightweight model integration | Nine fixed procedures; CPU/GPU synthetic and all45 whole-FIT logical admissions PASS;36 actual M8 and26 actual D1 cells identified | Finish production training,independently verify; measure compute alongside11690-parameter count |
| Performance | Old7 development matrix complete; integrated training active; no new discovery Outer inference yet | Freeze all trained models/maps/scalers/scales/thresholds;final26-procedure pass with no reselection;independent endpoint verification |
| Explanation | Source and measured floor/correlation diagnostics | Feature utility with paired controls and uncertainty,negative controls and quantitative cross-patient stability,not importance coefficients alone |
| Reproducibility/delivery | Source locks and private atomic caches | Independent verifiers,compact anonymous tables,final report and requirement-by-requirement completion audit |

Linear screening failure does **not** authorize skipping nonlinear M8 experiments or declaring a feature biologically useless. All future candidate/architecture choices must remain within each originalFIT, or be prospective outcome-free common specifications. Original validation only chooses final epoch/threshold under a locked rule. Do not form a global best feature set using validation averages and then call original Outer an independent test.

## Execution locations and continuation

Trusted server code: `D:\m8_physiology_feature_discovery_seed42_v1`.
Private runtime: `D:\m8_physiology_feature_discovery_seed42_v1_runtime`.
Safe SSH helper and original environment remain unchanged. Keep raw,identity-bearing records,models/predictions and logs on server. Do not use installed research/extraction skills or delegate to agents. Preserve previous branches/experiments and unrelated user changes.

Phases A/B/C have separate explicit source lists in their freeze bindings, so new follow-up tools can be added without rewriting those freezes. Do not change previously frozen files to accommodate later scripts. `OLD7_AUDIT_STATUS`, `CANDIDATE_EXTRACTION_STATUS`, and `CANDIDATE_SCREEN_STATUS` intentionally say the full goal is incomplete.

Next concrete work: separately lock the stable-log bandpower numerical controls motivated by the measured RMS/fallback issues; complete FIT-only center-held-out and nonlinear original-feature/candidate utility checks; then prospectively fix lightweight matched original/pruned/new-only/combined M8 variants (including an architecture/normalization control if input width changes) before original validation outcomes are consulted. Preserve matched Raw initialization,pre-reference,RawPR and budgets. Do not let sparse linear eligibility become an excuse to skip nonlinear/combined-feature tests. Record parameter/FLOP/runtime budgets and original88-position normalization geometry explicitly rather than swapping in a different LayerNorm unnoticed.

### Phase2 continuation update

Stable-log3 extraction and all1225 nonlinear patient/center screens are now actually complete and independently verified; anonymous artifacts are in `phase2_results`. `PHASE2_FINDINGS.md` distinguishes measured results from pending work. Per-FIT feature subsets and the nine integrated M8 procedures were frozen before reading the new neural-ablation validation aggregates. Synthetic94-position folding/initialization/Raw-prefix/gradient checks passed. `integrated_data.py` and `train_integrated.py` implement actual FIT-only transforms, availability-matched negative pairing, real whole-FIT admissions, unchanged original trainers and aliases; runtime admission/training completion is not inferred from the existence of these scripts.

The old7 matched matrix remains16×5 actual D1+M8 pairs. A sustained foreground execution wrapper waits for its verified terminal completion before launching integrated training, preventing duplicate or competing production GPU jobs. A prior silent SSH observation timeout was checked against the actual live PID; it did not stop training and did not justify a restart. `tools/verify_old7_training.py` is an independent no-refit validation replay verifier, not yet evidence until executed successfully.

Still required: finish/verify all old7 and integrated neural cells, freeze final models/maps/scales/thresholds, perform one fixed final repeated-test evaluation, verify all endpoint tables/uncertainty, audit compute and feature mechanism evidence, and complete the original requirement-by-requirement final report. The full goal remains active.

### Production continuation,2026-10-16

- Old7 matrix reached `ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE`:80 bases+80 M8 correction fits,16 procedures,5 folds. Development is65 appearances/47 unique validation patients, **not80-patient Outer**. Compact reports are in `phase3_results/old7_development`; independent full validation-forward replay is queued and not assumed passed.
- First integrated admission exposed a NumPy advanced-index axis bug in the wrong-pair negative control, before any formal integrated epoch. All original source/freezes/logs remain preserved. `ENGINEERING_PATCHES.md` documents the exact repair; the independent3×64×59×21 loop regression passed. No completed production epoch was discarded.
- Active production source is **`train_integrated_v2.py` + `integrated_data_v2.py`**, runtime **`integrated_M8_v2`**. Do not launch the initial v1 program or mutate either frozen source. All45 logical whole-FIT admissions passed;36 unique M8 inputs and26 unique D1 inputs are trained with unchanged original optimizers,budgets and selection. Source+both versions+regression are hash-bound.
- Whole-cohort negative pairing moves7633/7635 channels (99.9738%) within identical S×W validity-mask classes. Two singleton channels remain unchanged and are explicitly reported; no window/patient is dropped to force a permutation.
- Current sustained foreground trainer is PID24904 (revalidate live CIM/command/status before action), local tool session51642. A separate read-only wait/verification wrapper is in local session17574: it waits for that exact live trainer,requires the complete freeze,then runs old7 and integrated no-refit validation verifiers,stopping on any failure. Do not launch duplicates or infer a dead process from an observation timeout.
- `FINAL_EVALUATION_LOCK.json` and `evaluate_final.py` fix16 legacy-format ablations+9 integrated procedures+historical M8 (26 procedures/130 foldcells/2080 patient rows). They are **not yet executed**. They gate on both independent training verifiers,freeze models/maps/scalers/scales/cache sources/thresholds,seal all scores before full labels,and report primary comparisons without promoting secondary procedures. Final report,compute audit and completion audit remain pending.

The2026 first20s spectral study also motivates a future temporal-summary control. Original D1 aggregates all genuine source windows; compare old7 and new measurements under the same0–20s/early-late rule if that alternative is tested, rather than confounding a new feature with a different time support. Cache deltas already permit this without new EEG extraction. Lock the alternatives before efficacy analysis. Existing Raw-only/M8 representations must not be silently replaced or tuned against their old Outer scores.
