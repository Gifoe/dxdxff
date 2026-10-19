# Pruned single-head branch fusion: actual internal results

Exploratory repeated-test:33unique within-FIT ASSESS patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828 is unchanged and is not directly compared to these internal values.

## Fixed comparison

B0 is the original11425parameter structure. JOINT_PRUNED removes EarlyFiLM/lateGate with original jointGELU. FULL and itsPR/pre-reference controls use separate branchGELUs summed with fixed1/sqrt(2),then the same one64-hidden/one-scalar head. All6new variants9329parameters; no new hiddenPR or residualgain. Pruning changes capacity; fusion and individual PR/pre-reference contrasts match parameter count.

FeaturePR off retains both modalities and uses TRAIN-only global imputation/scaling. RawPR off retains rawsummaries with a frozen complete-TRAIN initial-encoder affine. These are not zeroed inputs or inference-only deletions. The affine does not track current encoder distribution. All controls were retrained from matched original initialization.

90fresh fits and15verified original internalB0reuses. Same original five paths,seeds42/43/44,complete patients/raw windows,loss/optimizer/budget andCAL-only selection. All105selected cells replayed/frozen and all105scores sealed before metrics; no source/threshold/arm/seed change from these outcomes.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.619955 | 0.752344 | 0.572237 | 0.408406 | 0.831505 | 0.752759 | 0.820934 | 0.729798 | -0.107229 |
| JOINT_PRUNED | 0.625490 | 0.753058 | 0.564077 | 0.421698 | 0.829283 | 0.753704 | 0.819592 | 0.727273 | -0.092279 |
| FULL | 0.616831 | 0.742477 | 0.553942 | 0.405604 | 0.828059 | 0.751264 | 0.822423 | 0.739899 | -0.100457 |
| WO_FEATURE_PR | 0.589476 | 0.714182 | 0.533424 | 0.344213 | 0.834740 | 0.751624 | 0.788245 | 0.681818 | -0.144475 |
| WO_RAW_PR | 0.571906 | 0.762723 | 0.579203 | 0.349863 | 0.793948 | 0.730176 | 0.858105 | 0.772727 | -0.066319 |
| WO_BOTH_PR | 0.486733 | 0.730725 | 0.547003 | 0.216787 | 0.756679 | 0.691464 | 0.806117 | 0.709596 | -0.058654 |
| WO_PRE_REFERENCE | 0.588467 | 0.685083 | 0.508734 | 0.360758 | 0.816176 | 0.730195 | 0.772454 | 0.661616 | -0.104723 |

Primary estimates average repeated folds within patient/seed,then all3seed metrics,then patients equally. Pooled appearances,fold/seedSD,source and coverage are separate. Sources<5patients are suppressed,not dropped.10000unique-patient paired bootstrap preserves all paths/seeds;135nominal and10predeclared additional intervals are conditional on historical selection and do not cover full training randomness.

## Fixed qualification

{"active_goal_achieved": false, "both_separate_PR_point_rules_pass": false, "fusion_point_rule_pass": false, "new_Outer_authorized": false, "performance_margin_pass": false, "point_rule_not_formal_inference": true, "pre_reference_point_rule_pass": true, "qualified_for_final_study_planning": false, "status": "FAIL"}

Positive seed directions alone do not establish statistical support; inspect paired intervals. OnlyFULL is the candidate. An apparently better control cannot be substituted post hoc. Passing point rules permits only planning final original-protocol work,not automatically readingOuter. Final80-patient Macro-F1>0.64,AUROC improvement and retained-module contributions remain unproven by this stage. Failed findings remain unchanged; no relaxed gates or weakened controls.

This simplification was motivated by the previously completed negative residual/core studies,not an independent validation. No biological causality or globally optimal feature set is claimed.

Only source,hashes and anonymous aggregates are public. Raw,identities,individual scores,anchors,preprocessors,weights,caches and runtime logs remain private.
