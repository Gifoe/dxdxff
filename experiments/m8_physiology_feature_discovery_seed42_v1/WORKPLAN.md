# M8 physiology feature discovery, seed42 — completion evidence and historical ledger

The full requested research scope has now been executed and independently verified,not reduced to data extraction or screening. The prospective primary improvement gate failed;diagnostic ranking gains and counterexamples to individual old-feature necessity are reported honestly. A completed bounded research result is not a claim of global7D optimality. `REPORT.md` and `GOAL_COMPLETION_AUDIT.json` define the final evidence state;older snapshots below remain for provenance.

Original objective: 以M8为基准，自主审计现有7D生理特征的必要性、冗余性与跨患者稳定性，结合最新研究探索更具SOZ定位价值的新特征。自主完成特征构建、筛选、消融、模型整合及性能验证，严格避免测试集信息泄漏。最终目标是在保持模型轻量化的同时提升AUROC、AP和F1，并通过可复现实验明确哪些特征真正有效、为何有效，以及是否优于原始7D；如果能找到需要依据，找不到需要证明现有的合理性。

## Initial execution (historical stage)

- Revalidated original code/data provenance, patient-held-out partitions and exact seven-feature window registry. No new Outer outcome read.
- Targeted primary-literature review includes2026 multi-band peri-onset localization and2024 complexity/spectral-decomposition studies; separate clinical targets and unsupported signal requirements are identified in `LITERATURE_EVIDENCE.md`.
- Locked11 new cache-supported feature formulas before extraction. Analytic/scale/constant/mask/singleton unit tests passed.
- Ran225 original7 linear subFIT/calibration/screen probes (full,seven single groups,seven deletions) inside originalFIT only; no original validation labels used for feature screening.
- Label-blind extraction actually completed for80patients/7635channels/256seizures/1471965valid channel-windows, finite100%; existing actual7D window deltas reaggregate exactly to R14.
- Ran390 prospective candidate probes inside the same originalFIT screens. Retained private classifiers,scalers,probabilities,labels and indices for independent verification. No global feature winner selected across original validation or Outer.
- Independent verifier replayed all390 candidate probability/threshold pipelines and confirmed all80 candidate caches against original aggregation; max replay and aggregation errors0. Only compact anonymous tables were fetched into `initial_results`.

## Final requirement evidence

| Requirement | Final evidence | Interpretation boundary |
|---|---|---|
| Old7 necessity | All80 D1+M8 pairs independently replayed;all16 M8 procedures tested on5 Outer folds | Conditional ablations,not proof each physiological measurement is essential |
| Redundancy | Per-FIT Spearman/fallback/floor audits plus actual single/drop/full retraining and94-position control | Distinguish correlation,algebra,normalization geometry and utility;deletion counterexamples reported |
| Cross-patient/source stability | All705 patient and520 original-center-label holdouts independently verified,orientation and uncertainty included | Four originalsource labels,not necessarilyfour physicalhospitals;smallgroup and repeatedcohort limits |
| New feature construction | Full-cohort11 new+3 numerical controls,formula/mask/unit tests,exact oldR14 replay andhash-bound caches | No unavailable LF/fullHFO/PLHG or unmeasured biomarker claims |
| Feature selection | All5 FIT-specific lists independently reproduced andfrozen before new neural efficacy aggregates | No global validation/Outer-selected feature winner;diagnostic routes not promoted |
| Lightweight integration | All45 logical procedures complete;36 actualM8+26 actualD1;allsynthetic andrealFIT admissionsPASS;11690params andmeasured shape/compute |9 aliases are exactreuse,not extra fits;affine counts are not fullFLOPs;offline featurecost included |
| Performance | One frozen130-cell prediction pass for26procedures;2080patient rows,allscore/input/cache/metric/bootstrap checksPASS |Primary synchronous-improvement gate failed;allresults exploratory repeated-test,not blindconfirmation |
| Explanation | D1-only/gate-only/new-only/numeric/wrong-pair controls,paired intervals andall old7 counterexamples inREPORT |Ranking signal does not establish F1 gain or causal SOZ biology;no global optimum proof |
| Reproducibility/delivery | All source locks,32 finalanonymousartifacts,4 publicstageverifiers,completionverifier,REPORT/README andfull evidenceledger |No raw/identities/predictions/checkpoints/scalers/logs released;externalpush not claimed |

Linear screening failure does **not** authorize skipping nonlinear M8 experiments or declaring a feature biologically useless. All future candidate/architecture choices must remain within each originalFIT, or be prospective outcome-free common specifications. Original validation only chooses final epoch/threshold under a locked rule. Do not form a global best feature set using validation averages and then call original Outer an independent test.

## Execution locations and continuation

Trusted server code: `D:\m8_physiology_feature_discovery_seed42_v1`.
Private runtime: `D:\m8_physiology_feature_discovery_seed42_v1_runtime`.
Safe SSH helper and original environment remain unchanged. Keep raw,identity-bearing records,models/predictions and logs on server. Do not use installed research/extraction skills or delegate to agents. Preserve previous branches/experiments and unrelated user changes.

Phases A/B/C have separate explicit source lists in their freeze bindings, so new follow-up tools can be added without rewriting those freezes. Do not change previously frozen files to accommodate later scripts. `OLD7_AUDIT_STATUS`, `CANDIDATE_EXTRACTION_STATUS`, and `CANDIDATE_SCREEN_STATUS` intentionally say the full goal is incomplete.

Historical next-action note (now completed): separately lock stable-log numerical controls;complete FIT-only center-held-out/nonlinear checks;fix matched original/pruned/new-only/combined M8 variants before validation aggregates. Preserve matched Raw initialization,pre-reference,RawPR and budgets;record normalization and compute explicitly. All these stages are now evidenced above rather than inferred from a plan.

### Phase2 continuation update (historical snapshot, superseded by final evidence)

Stable-log3 extraction and all1225 nonlinear patient/center screens are now actually complete and independently verified; anonymous artifacts are in `phase2_results`. `PHASE2_FINDINGS.md` distinguishes measured results from pending work. Per-FIT feature subsets and the nine integrated M8 procedures were frozen before reading the new neural-ablation validation aggregates. Synthetic94-position folding/initialization/Raw-prefix/gradient checks passed. `integrated_data.py` and `train_integrated.py` implement actual FIT-only transforms, availability-matched negative pairing, real whole-FIT admissions, unchanged original trainers and aliases; runtime admission/training completion is not inferred from the existence of these scripts.

The old7 matched matrix remains16×5 actual D1+M8 pairs. A sustained foreground execution wrapper waits for its verified terminal completion before launching integrated training, preventing duplicate or competing production GPU jobs. A prior silent SSH observation timeout was checked against the actual live PID; it did not stop training and did not justify a restart. `tools/verify_old7_training.py` is an independent no-refit validation replay verifier, not yet evidence until executed successfully.

Still required: finish/verify all old7 and integrated neural cells, freeze final models/maps/scales/thresholds, perform one fixed final repeated-test evaluation, verify all endpoint tables/uncertainty, audit compute and feature mechanism evidence, and complete the original requirement-by-requirement final report. The full goal remains active.

### Production continuation,2026-10-16 (historical snapshot; listed processes are now terminal)

- Old7 matrix reached `ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE`:80 bases+80 M8 correction fits,16 procedures,5 folds. Development is65 appearances/47 unique validation patients, **not80-patient Outer**. Compact reports are in `phase3_results/old7_development`; independent full validation-forward replay is queued and not assumed passed.
- First integrated admission exposed a NumPy advanced-index axis bug in the wrong-pair negative control, before any formal integrated epoch. All original source/freezes/logs remain preserved. `ENGINEERING_PATCHES.md` documents the exact repair; the independent3×64×59×21 loop regression passed. No completed production epoch was discarded.
- Active production source is **`train_integrated_v2.py` + `integrated_data_v2.py`**, runtime **`integrated_M8_v2`**. Do not launch the initial v1 program or mutate either frozen source. All45 logical whole-FIT admissions passed;36 unique M8 inputs and26 unique D1 inputs are trained with unchanged original optimizers,budgets and selection. Source+both versions+regression are hash-bound.
- Whole-cohort negative pairing moves7633/7635 channels (99.9738%) within identical S×W validity-mask classes. Two singleton channels remain unchanged and are explicitly reported; no window/patient is dropped to force a permutation.
- Current sustained foreground trainer is PID24904 (revalidate live CIM/command/status before action), local tool session51642. A separate read-only wait/verification wrapper is in local session17574: it waits for that exact live trainer,requires the complete freeze,then runs old7 and integrated no-refit validation verifiers,stopping on any failure. Do not launch duplicates or infer a dead process from an observation timeout.
- `FINAL_EVALUATION_LOCK.json` and `evaluate_final.py` fix16 legacy-format ablations+9 integrated procedures+historical M8 (26 procedures/130 foldcells/2080 patient rows). They are **not yet executed**. They gate on both independent training verifiers,freeze models/maps/scalers/scales/cache sources/thresholds,seal all scores before full labels,and report primary comparisons without promoting secondary procedures. Final report,compute audit and completion audit remain pending.

The2026 first20s spectral study also motivates a future temporal-summary control. Original D1 aggregates all genuine source windows; compare old7 and new measurements under the same0–20s/early-late rule if that alternative is tested, rather than confounding a new feature with a different time support. Cache deltas already permit this without new EEG extraction. Lock the alternatives before efficacy analysis. Existing Raw-only/M8 representations must not be silently replaced or tuned against their old Outer scores.

### Final current state

All222 actual neural fits,all26 final five-fold procedures andindependent verifiers are complete. Original160-model validation replay maximumerror0;integrated validation replay max4.7684e-7;finalcache input reconstruction andall2080 patient metrics/rankings,pooled/fold/source tables and33paired10000-draw bootstrap comparisonsPASS. Scalers,models,featuremaps,gate scales andthresholds were frozen before inference;all130scores were sealed before full labels. No post-Outer training/tuning occurred. Reporting-only native failures were recovered using unchanged frozen statistical AST in a parity-gated CPU process;no neural prediction was repeated. `ENGINEERING_PATCHES.md` describes uncertainty about the lower-level native cause.

The optional temporal-summary idea above was not run andis not claimed as an achieved experiment. It is outside this completed fixed candidate study,not a reason to reinterpret the current finite evidence as exhaustive. The original research objective is retained;the delivered answer is mixed/negative for its predeclared primary gate,with bounded positive ranking evidence and explicit weaknesses of old7.
