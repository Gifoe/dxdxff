# Fixed simplification and branch-fusion comparison

Mode: design/execution, not manuscript or a result claim. All new outcomes are pending.

The completed residual study failed against the old structure and a matched context control. Its core audit did not support EarlyFiLM/correct-donor specificity, while a pre-ictal reference had appreciable AUROC evidence. Removing both guidance operations had higher AUROC/Macro points but uncertain paired differences and lower AP. These observations motivate this new, explicitly exploratory comparison; they do not prove that pruning works or that physiology is uninformative. Prior failed gates are not changed.

## One candidate, two structural controls

| Arm | Fixed role | New fits |
| --- | --- | ---: |
| B0 | exact original six-feature structure, internal reference | 0;15 verified reuses |
| JOINT_PRUNED | remove both guidance operations; retain original joint GELU | 15 |
| FULL | same pruning, branch-separate GELUs summed before one scalar head | 15 |
| WO_FEATURE_PR | FULL without physiology patientZ; TRAIN-only global preprocessing | 15 |
| WO_RAW_PR | FULL without rawsummary patientZ; TRAIN-only initial affine | 15 |
| WO_BOTH_PR | FULL without both patientZ operations | 15 |
| WO_PRE_REFERENCE | FULL without own-channel pre-ictal latent subtraction | 15 |

All six new arms have9329parameters, versus11425for B0. The fusion contrast matches capacity; the pruning contrast necessarily changes it. No added hidden PR, residual gain, attention, learned weighting or second classifier. Both modalities remain usable in all PR controls, and all controls are independently trained. One64-wide hidden representation and the original final output layer remain; split fusion uses the same weight slices, two parameter-free GELU calls, and fixed scaling.

## Evidence and acceptance

Freeze the executable protocol before new fitting. Use the same seeds42/43/44, all five original within-FIT paths, complete patients/windows, legal CAL-only epoch/threshold choice and original optimization budgets. Only one candidateFULL is tested; JOINT_PRUNED cannot be promoted post hoc ifFULL fails. Full selection replay, all-score sealing,840row/7560endpoint independent verification,135nominal and10additional interval checks are required. All nine endpoints, seed/fold/source/pooled and coverage summaries are mandatory.

The predeclared point qualification requires improved FULL AUROC versusB0 with at most1pp mean AP/Macro/EZF1 loss and consistent seeds, plus positive AUROC/Macro direction for each PR, retained pre-reference and fusion contrast. Statistical interval support is reported separately, never inferred from seed-direction counts. A pass permits only planning original-protocol final evaluation, not automaticOuter or goal completion. A failure is kept, not corrected by weakening controls.

The full active objective remains final80-patient fivefold Macro-F1>0.64, improved AUROC and reproducible retained-module contributions. No internal proxy score or documentation-only artifact satisfies it. HistoricalOuter0.653828 is unchanged. All historical-cohort findings remain exploratory repeated-test.

## Unique admission scope

Check only changed critical paths: exact pruned-joint wrapper versus independently pruned original logits/loss/gradients/Adam; exact shared initialization/RNG; split formula and both modality gradients; PR/affine and pre-reference switches; removed Variance/guidance isolation. Verify15B0reuse bridges and15new complete-TRAIN anchors. Admission updates are discarded; no synthetic metric is published as evidence.

Only code,hashes and anonymous aggregate outputs are public. No new extraction, Raw copy, private score/identity/weight/cache/log release or unrelated process change.
