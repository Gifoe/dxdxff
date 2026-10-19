# One Raw global-plus-relative candidate with matched contexts

Mode: experiment design/execution,not manuscript prose. All new outcome cells are pending.

The completed pruned/split-activation study failed its fixed rules. RawPR increased Macro-F1 but had negative AUROC points in all3seeds; splitting GELUs also failed against joint activation. Those actual negative findings are kept. This study restores jointGELU and asks whether retaining TRAIN-global latent information alongside patient-relative information can avoid that trade-off. It does not assume either view helps.

## Frozen matrix

| Arm | Mechanism or reference | Fits |
| --- | --- | ---: |
| B0 | original six-feature structure,old internal control |15exact reuses |
| PRUNED_REF | parentJOINT_PRUNED control,not a promoted main |15exact reuses |
| FULL | global96D plus patient-relative96D Raw,14wide physiology,jointGELU |15fresh |
| WO_FEATURE_PR | no physiology patientZ;TRAIN-only global preprocessing |15fresh |
| WO_RAW_PR | retain globalRaw and replace patientRawZ by channelZ(globalRaw) |15fresh |
| WO_BOTH_PR | combine bothPRremovals |15fresh |
| WO_GLOBAL_VIEW | replace globalRaw by channelZ(relativeRaw),keep relative view |15fresh |
| WO_PRE_REFERENCE | skip own-channel pre-reference;fit its own TRAIN-only scale |15fresh |

All new models have15473parameters and the same192D Raw input,one64-hidden head and one scalar output. RawPR/globalview controls retain valid information and a nonlinear within-channel alternative,not zeros. These match parameter count,not necessarily function-class expressiveness. References have fewer parameters; conditional matched controls are needed before attributing gains to context rather than capacity.

The global view is a pre-patientZ latent summary standardized using initialTRAIN statistics. It does not restore physical amplitude removed by original per-window normalization. Independent pre/no-pre TRAIN anchors avoid a scale-disadvantaged pre-reference deletion; anchors remain fixed,not current-encoder estimates. No extra hiddenPR layer or old unsupported gains/guidance are retained.

## Evidence contract

Same original80patients/7635channels/256seizures/1471965valid raw250windows and five paths,seeds42/43/44,within-FIT TRAIN34–36/CAL8/ASSESS8.90fresh fits and30exact references,all120selections freeze and all120scores seal before metrics. Report all9patient endpoints,source/fold/seed/pooled/coverage;independently verify960rows/8640values,144nominal and12additional intervals. Primary context comparisons require both AUROC/Macro direction and seed consistency;paired intervals remain separate. The fixed point gate permits only planning a final original-protocol study,not automaticOuter.

OnlyFULL is the new candidate. A better-looking control cannot replace it after results. Final80-patient Macro-F1>0.64,AUROC improvement and statistically stable retained-module contributions remain actual requirements;this internal study alone cannot achieve them. All historical-cohort analysis is exploratory repeated-test.

## Minimal changed-path admission

Check shared tensors/RNG/headlayout,independent raw-pooling/dual-view forward-and-gradient parity,referenceCAL reuse,scope/invariance of patient versus within-channel context,maskedVariance/removedguidance isolation and matched pre/no-pre global anchors. Use real completeTRAIN patients for finite gradients and oneAdam-state parity;discard admission updates. No repetitive toy accuracy comparisons.

No raw copying,new extraction,identity/score/checkpoint/cache/log release or unrelated process mutation. Source/protocol go toGitHub before formal fits;only anonymous aggregates follow. Previous failed qualifications and model records stay unchanged.
