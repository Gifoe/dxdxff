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
| Old7 necessity | Linear single/drop probes only | Matched nonlinear original-D1 and full-M8 feature deletion/single/pathway ablations; patient-isolated checkpoint/threshold selection |
| Redundancy | FIT-only Spearman and numeric-scale tables | Distinguish algebraic duplication,correlation,conditional utility and normalization geometry; verify any pruning by matched retraining |
| Cross-patient/center stability | Patient-isolated screens and center-stratified summaries | Explicit leave-center-out sensitivity inside FIT,orientation consistency and uncertainty; no pooled-center shortcut |
| New feature construction | Actual11-candidate complete extraction | Independent source/cache/math checks and end-to-end numerical gates; no LF/HFO claims outside source support |
| Feature selection | Provisional per-FIT linear eligibility only | Nonlinear/center confirmation and per-fold frozen subsets; combined feature checks with honest selection accounting |
| Lightweight model integration | Not implemented | Fix original/pruned/new-only/combined matched M8 variants before reading original validation outcomes; verify parameters,compute,initialization and gradients |
| Performance | No candidate M8 validation/Outer inference yet | Complete original five validation folds,register final selection,freeze scalers/models/thresholds,then at most one declared final repeated-test evaluation |
| Explanation | Source and measured floor/correlation diagnostics | Feature utility with paired controls and uncertainty,negative controls and quantitative cross-patient stability,not importance coefficients alone |
| Reproducibility/delivery | Source locks and private atomic caches | Independent verifiers,compact anonymous tables,final report and requirement-by-requirement completion audit |

Linear screening failure does **not** authorize skipping nonlinear M8 experiments or declaring a feature biologically useless. All future candidate/architecture choices must remain within each originalFIT, or be prospective outcome-free common specifications. Original validation only chooses final epoch/threshold under a locked rule. Do not form a global best feature set using validation averages and then call original Outer an independent test.

## Execution locations and continuation

Trusted server code: `D:\m8_physiology_feature_discovery_seed42_v1`.
Private runtime: `D:\m8_physiology_feature_discovery_seed42_v1_runtime`.
Safe SSH helper and original environment remain unchanged. Keep raw,identity-bearing records,models/predictions and logs on server. Do not use installed research/extraction skills or delegate to agents. Preserve previous branches/experiments and unrelated user changes.

Phases A/B/C have separate explicit source lists in their freeze bindings, so new follow-up tools can be added without rewriting those freezes. Do not change previously frozen files to accommodate later scripts. `OLD7_AUDIT_STATUS`, `CANDIDATE_EXTRACTION_STATUS`, and `CANDIDATE_SCREEN_STATUS` intentionally say the full goal is incomplete.

Next concrete work: separately lock the stable-log bandpower numerical controls motivated by the measured RMS/fallback issues; complete FIT-only center-held-out and nonlinear original-feature/candidate utility checks; then prospectively fix lightweight matched original/pruned/new-only/combined M8 variants (including an architecture/normalization control if input width changes) before original validation outcomes are consulted. Preserve matched Raw initialization,pre-reference,RawPR and budgets. Do not let sparse linear eligibility become an excuse to skip nonlinear/combined-feature tests. Record parameter/FLOP/runtime budgets and original88-position normalization geometry explicitly rather than swapping in a different LayerNorm unnoticed.

The2026 first20s spectral study also motivates a future temporal-summary control. Original D1 aggregates all genuine source windows; compare old7 and new measurements under the same0–20s/early-late rule if that alternative is tested, rather than confounding a new feature with a different time support. Cache deltas already permit this without new EEG extraction. Lock the alternatives before efficacy analysis. Existing Raw-only/M8 representations must not be silently replaced or tuned against their old Outer scores.
