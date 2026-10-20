# Global-plus-relative Raw dual view: actual internal results

Exploratory repeated-test:33unique within-original-FIT ASSESS patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828 remains unchanged and is not directly subtracted from these internal values.

## Fixed architecture and controls

B0andPRUNED_REF are exact frozen parent references,not newly selected winners.90fresh fits and30reference reuses,same original five paths,seeds42/43/44,full patients/windows/loss/optimization/CAL-only selection. All120selections replayed/frozen and all120score cells sealed before metrics.

Each new arm has15473parameters,one64-hidden jointGELU and one scalar classifier. The original unconditioned24D CNN/pre-reference/stage/seizure pooling gives96D v. FULL concatenates TRAIN-global affine g(v) and current patientZ(v),then the same six-feature14wide masked summary. EarlyFiLM,lateGate,old residualgains and extra hiddenPR remain absent. B0has11425parameters andPRUNED_REF9329; unconditional reference improvements alone cannot establish context-specific benefit.

RawPR deletion retains g and substitutes channelZ(g) for patientZ(v). Global-view deletion retains r=patientZ(v) and substitutes channelZ(r) for g. These are parameter-/dimension-matched nonzero within-channel alternatives,not necessarily identical function classes. FeaturePR deletion uses TRAIN-only global imputation/scaling,not deleted physiology. BothPR deletion combines them while retaining per-window waveform normalization and own-channel temporal reference.

The global view is a latent summary before final patientZ,not restored physical EEG amplitude. Its fixed initialTRAIN affine does not track the evolving encoder.30new pre/no-pre anchors were independently fitted and replayed on all genuine TRAIN windows without labels orCAL/ASSESS. The no-pre control fits its own legal scale; its contrast includes that necessary preprocessing refit,not only an isolated algebraic subtraction.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.619955 | 0.752344 | 0.572237 | 0.408406 | 0.831505 | 0.752759 | 0.820934 | 0.729798 | -0.107229 |
| PRUNED_REF | 0.625490 | 0.753058 | 0.564077 | 0.421698 | 0.829283 | 0.753704 | 0.819592 | 0.727273 | -0.092279 |
| FULL | 0.600308 | 0.762056 | 0.585878 | 0.379927 | 0.820689 | 0.749149 | 0.873770 | 0.815657 | -0.102311 |
| WO_FEATURE_PR | 0.560582 | 0.722467 | 0.545433 | 0.308427 | 0.812737 | 0.730715 | 0.774696 | 0.664141 | -0.120805 |
| WO_RAW_PR | 0.564538 | 0.755792 | 0.570848 | 0.345579 | 0.783497 | 0.717222 | 0.851064 | 0.762626 | -0.058138 |
| WO_BOTH_PR | 0.488972 | 0.716051 | 0.530192 | 0.234180 | 0.743763 | 0.679218 | 0.780759 | 0.659091 | -0.046212 |
| WO_GLOBAL_VIEW | 0.618986 | 0.743937 | 0.550959 | 0.410410 | 0.827562 | 0.750492 | 0.821254 | 0.729798 | -0.093726 |
| WO_PRE_REFERENCE | 0.586042 | 0.696265 | 0.520811 | 0.346313 | 0.825771 | 0.742650 | 0.793091 | 0.702020 | -0.127098 |

Primary estimates average repeated folds within patient/seed,then all3seed metrics,then patients equally. Source<5suppressed,not dropped;pooled appearances,seed/foldSD and coverage are separate.10000unique-patient paired bootstrap retains all paths/seeds.144nominal and12fixed additional intervals are conditional on historical research selection and do not cover full training randomness.

## Fixed next-stage rule

{"active_goal_achieved": false, "both_separate_PR_point_rules_pass": true, "expanded_reference_point_rule_pass": false, "global_view_point_rule_pass": false, "new_Outer_authorized": false, "performance_margin_pass": false, "point_rule_not_formal_inference": true, "pre_reference_point_rule_pass": true, "qualified_for_final_study_planning": false, "status": "FAIL"}

Seed-direction consistency is not statistical proof:inspect nominal/additional intervals separately. OnlyFULL is the candidate. A better-looking control cannot be promoted post hoc. A point pass permits only planning final original-protocol work,not automatic newOuter or active-goal completion. Final80-patient Macro-F1>0.64,improvedAUROC and stable retained-module contributions still require actual evidence. Prior failed studies and their gates remain unchanged.

No newOutcome-driven source,threshold,cohort,seed or comparison-family changes. No independent-confirmation,biological-causality or global-optimality claim.

Only source,hashes and anonymous aggregates are public. Raw,identities,individual scores,anchors,preprocessors,weights,caches and runtime logs remain private.
