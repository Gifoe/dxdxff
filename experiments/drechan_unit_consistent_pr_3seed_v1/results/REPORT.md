# Unit-consistent FeaturePR:actual internal experiment

Exploratory repeated-test:33unique within-original-FIT assessment patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828remains unchanged,not a guaranteed post-correction score or a comparable internal endpoint.

## Exact scope

90fresh fits plus45exact reference reuses under original five roles,seeds42/43/44,full patients/windows/loss/AdamW/precision/budgets andCAL-only epoch/threshold selection. All135selections replayed/frozen andall135score cells sealed before unified metrics.

Only direct FeaturePR numerical definition changes. TRAIN-only raw imputer/StandardScaler provides per-coordinate units. Patient SD is compared with1e-8times that TRAINunit;valid coordinates use centered/rawSD anddegenerate observed coordinates become0,missing values stayNaN. The original second imputer/StandardScaler is fitted onTRAIN only. All6physiological features remain,with only the oldVariance pair masked. No clipping,patient exclusion,feature search oroutcome-tuned epsilon.

The NoFeaturePR matrix/scaler is the original TRAIN-only global input,bit unchanged. AllNoFeaturePR/bothPR models were nevertheless freshly retrained andindependently required to match the prior identical tensor procedure. A fair control is not made weaker by a new scale or missing branch.

The dual-view CNN/96Dpooling/192DRawinput/14widephysiology/one64-hidden jointGELU/scalar head,RawPR,pre-reference andfixed initialTRAIN Rawanchors are unchanged. No evolving-encoder anchor refresh,loss ordecoder rescue is added. All6new arms have15473parameters;referenceB0has11425,PRUNED_REF9329,andLEGACY_DUALVIEW15473. Reference-only gains cannot isolate added capacity;matched module controls are separately reported.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.619955 | 0.752344 | 0.572237 | 0.408406 | 0.831505 | 0.752759 | 0.820934 | 0.729798 | -0.107229 |
| PRUNED_REF | 0.625490 | 0.753058 | 0.564077 | 0.421698 | 0.829283 | 0.753704 | 0.819592 | 0.727273 | -0.092279 |
| LEGACY_DUALVIEW | 0.600308 | 0.762056 | 0.585878 | 0.379927 | 0.820689 | 0.749149 | 0.873770 | 0.815657 | -0.102311 |
| FULL | 0.608334 | 0.753474 | 0.566453 | 0.389118 | 0.827549 | 0.753101 | 0.835765 | 0.757576 | -0.102233 |
| WO_FEATURE_PR | 0.560582 | 0.722467 | 0.545433 | 0.308427 | 0.812737 | 0.730715 | 0.774696 | 0.664141 | -0.120805 |
| WO_RAW_PR | 0.578824 | 0.756865 | 0.572216 | 0.357372 | 0.800275 | 0.729500 | 0.860071 | 0.780303 | -0.073724 |
| WO_BOTH_PR | 0.488972 | 0.716051 | 0.530192 | 0.234180 | 0.743763 | 0.679218 | 0.780759 | 0.659091 | -0.046212 |
| WO_GLOBAL_VIEW | 0.618676 | 0.748194 | 0.553038 | 0.409292 | 0.828061 | 0.751452 | 0.806550 | 0.712121 | -0.087885 |
| WO_PRE_REFERENCE | 0.588933 | 0.696189 | 0.522409 | 0.353849 | 0.824017 | 0.741503 | 0.797066 | 0.704545 | -0.120016 |

Primary estimates average repeated folds within patient/seed,then3seed metrics,then patients equally. Separate pooled,source/fold/seedSD andcoverage tables are provided. Source<5unique patients suppressed,not dropped.10000unique-patient seed42paired bootstrap gives162nominal and14predeclared additional-family intervals,conditional on models andhistorically adaptive research,not independent confirmation or full training uncertainty.

## Fixed qualification

{"active_goal_achieved": false, "both_separate_PR_point_rules_pass": false, "expanded_reference_point_rule_pass": false, "global_view_point_rule_pass": false, "new_Outer_authorized": false, "performance_margin_pass": false, "point_rule_not_formal_inference": true, "pre_reference_point_rule_pass": true, "qualified_for_final_study_planning": false, "status": "FAIL", "unit_consistency_point_rule_pass": false}

Seed directions andpoint rules do not establish statistical stability. Qualification requires the main performance andretained-module point rules;UNIT_CONSISTENCY against the old mixed-unit procedure is reported as a diagnostic comparison,not proof that a mathematical correction must improve prediction. Its nominal/additional intervals are not omitted on a negative result. No better control is promoted after outcomes.

Internal completion orqualification does not achieve final80-patient Macro-F1>0.64,improvedAUROC orstable retained-module contributions anddoes not automatically authorize newOuter. Actual final original-protocol fitting/freeze/evaluation remains required. No model,threshold,comparison family,patient orseed changed after newassessment.

Only code,hashes andanonymous aggregates are public. Raw,identities,individual predictions,weights,preprocessors,anchors,caches andlogs remain on the trusted server.
