# Mechanism audit: inner evidence versus user-authorized Outer

All analyses use fixed predictions and numeric thresholds. Context is query-excluded, same-time and same-seizure; its genuinely achieved disruption passed before fitting. No biological normal reference or propagation causality is identified.

## FIT-only inner audit

C1-C0 AUROC +0.068110 [0.043811,0.093032], C1-C2 +0.026080 [0.010009,0.044364]. C2-C0 also +0.042030 [0.023449,0.060687]. Thus generic contextual summaries explain a substantial part, but not all, of the inner contrast. All5 original folds and all4 known-center AUROC point comparisons C1-C0 and C1-C2 are positive. C1-C2 improves141/255 appearances, worsens108, ties6—not141 independent people.

Ranks change: C1-C0 mean Spearman0.714708, mean strict pair reversal fraction0.209344; top1 channel identity changes38.82%. C1-C2 mean Spearman0.834424, strict reversal0.152787. These threshold-free changes reject a thresholds-only explanation under the fitted probe, but do not imply useful top-ranked EZ retrieval: MRR/Top1 paired intervals include zero.

Repeated appearances, not unique-channel counts: versusC0, C1 recovers628 EZ true positives but spoils573; adds1050 FPs but removes2423. Net TP+55,FP-1373. VersusC2: TP+179,FP+208. Patient-equal sensitivity +0.026520 versusC0 has CI crossing zero; specificity +0.088331 has positive CI. The F1 increase versusC0 is substantially compatible with improved specificity, not just more EZ detections.

Frozen C1 early-only shifted AUROC0.698164, later-only shifted0.676964; fullC1 minus interventions +0.016721 [0.004618,0.034003] and +0.037921 [0.018792,0.060205]. Both stage correspondences contribute in this fixed inner representation; no stage was selected or retrained. Setting standardized context coordinates0 in the **same C1 model** gives AUROC0.494184. This is a co-adaptation/intervention diagnostic, NOT the independently fitted C0 baseline or proof context contains unique biological information.

Channel count: C1-C2 AUROC point gains <=64:+0.084543,65-128:+0.004071,>128:+0.059142; no monotonic more-channels-only pattern. Prespecified OLS contrast at64channels/2seizures/equal-fold composition: C1-C0 +0.083566 [0.034066,0.132981]; C1-C2 +0.052179 [0.011962,0.094096]. Descriptive linear adjustment, not causal or an independent replication.

Seizure strata C1-C2 AUROC: one:+0.123049,two:-0.002936,three-plus:+0.022897. The advantage is not restricted to multiple seizures and is not uniform: the two-seizure group fails the point comparison. No subgroup was selected.

## Outer weakens synchronization-specific attribution

Separate final15-probe FIT training and TRAIN-only threshold-selection splits were locked after user authorization, before Outer prediction/label access. Outer covers80 unique people once,7,635 channels. It is exploratory/repeated Outer because this cohort has extensive prior use and the extension followed the inner result.

C1-C0 AUROC +0.039195 [0.002149,0.077865], AP+0.027319 [0.001728,0.054562], Macro-F1+0.066680 [0.036374,0.097638]. However C1-C2 AUROC +0.013444 [-0.006018,0.034423] and AP+0.001517 [-0.013195,0.016250]. AUROC improves39 people and worsens39, ties2. This does not pass synchronization-specific replication.

C1-C2 Macro-F1 +0.016644 [0.000828,0.033772], but EZ-F1 +0.016499 has CI crossing zero. MRR falls0.041028 and Top1 falls0.0625 compared withC2 (their intervals also cross zero). Shifted C2 has the highest Outer MRR0.726676 and Top1 0.6500. Do not omit this contrary ranking evidence.

Outer C1-C0 mean rank Spearman0.698834 and pair reversal0.216772; top1 channel changes51.25%. C1-C2 reversal0.148289, top1 identity changes32.5%. Consequently C1 alters channel ordering, not just thresholds—but ranking alteration is not uniformly beneficial.

Outer unique-channel counts versusC0:170 TP recovered,260 spoiled (netTP-90);262 FP added,786 removed (netFP-524). VersusC2:86 TP recovered,84 spoiled (netTP+2);132 FP added,251 removed (netFP-119). C1's patient-equal sensitivity0.422034 is slightly aboveC0 0.416197, but pooled TP609 is belowC0 699 because patients/channel counts and EZ burdens differ. Do not conflate these estimators.

Known-center Outer AUROC C1-C0: HUP+0.067584,LZU+0.034490,Multicenter+0.029009,Pediatric-0.057111. C1-C2: +0.024443,+0.028569,+0.010071,-0.069425. Original fold4 is also negative againstC0 andC2. No unseen-center generalization is tested. No Outer stage intervention/channel-adjustment extension was searched after outcomes.

## Conclusion

The inner audit supports the specific probe, but the requested Outer extension fails its mandatory positive C1-C2 AUROC interval gate. Combined conclusion: **PATIENT_CONTEXT_POSSIBLY_USEFUL_SYNCHRONIZATION_NOT_IDENTIFIED**. Generic contextual variables remain a plausible explanation for much of the gain. No automatic TIDE design/training, rescue or additional seed is justified by the completed evidence. This does not establish that all cross-channel physiology or nonlinear synchronization models are useless.
