# P1/P2/A2: actual seed42 five-fold findings

All25cells are complete (15fresh fits/182epochs and10strict reuses). Every group contains the same80held-out patients/7635channels/256seizures/1471965validRaw250windows. Patient-equal metrics,all9endpoints,pooled/folds/sources/coverage and paired intervals are independently verified. The fixed no-Late main still excludes Variance and Head Interaction. Historical Outer viewing makes this exploratory repeated-test,not independent confirmation. No result changes the main or thresholds.

## All nine primary patient summaries

| Group | Parameters | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Full | 10377 | 0.653465 | 0.765286 | 0.593488 | 0.464629 | 0.842301 | 0.772424 | 0.846368 | 0.800000 | -0.034239 |
| Original noEarly | 9329 | 0.644071 | 0.766014 | 0.568578 | 0.448547 | 0.839594 | 0.764299 | 0.779291 | 0.675000 | -0.035998 |
| P1 temporal statistics | 10377 | 0.650744 | 0.773571 | 0.583444 | 0.472280 | 0.829208 | 0.754780 | 0.816476 | 0.750000 | 0.007175 |
| P2 time-aware attention | 10403 | 0.653678 | 0.768380 | 0.595115 | 0.464480 | 0.842876 | 0.773263 | 0.834055 | 0.775000 | -0.035619 |
| A2 capacity-matched noEarly | 10337 | 0.650836 | 0.774698 | 0.577704 | 0.459176 | 0.842495 | 0.768851 | 0.803202 | 0.700000 | -0.035785 |

EZ-fraction bias is predicted minus true EZ fraction; closeness to zero is desirable. [Machine-readable table](public/MAIN_RESULTS.csv). All groups retain the same physiological direct-input processing; P1/P2 change Raw aggregation only. A2 deletes physiological CNN modulation,not the direct physiology input.

## P1: is the original Raw temporal partition necessary?

P1 replaces0-8s/8-20s Raw stage means with0-20s valid-window population Mean/SD,then equal-seizure Mean/SD96D. It keeps EarlyFiLM,own pre-reference,RawPR,directFeaturePR and the original64-head.

Relative to Full,P1 changes Macro-F1 by **-0.002721**,AUROC **+0.008284**,AP **-0.010044**,EZ-F1 **+0.007651**,MRR **-0.029892**,Top-1 **-0.050000**. P1-minus-Full paired nominal95% intervals are[-0.016851,0.010536] for Macro-F1 and[-0.008745,0.027325] for AUROC. The original partition is not demonstrated to be indispensable by these results. Conversely,an interval spanning zero is not proof of equivalence or noninferiority; no such margin was predeclared.

P1 changes both temporal partition and the within-seizure statistic (introducing temporal SD). It cannot isolate segmentation alone or show that all temporal information/pre-reference is unnecessary. Its Macro-F1/AUROC fold sample SDs are0.043211/0.067290 versus Full0.033102/0.046812,so the higher AUROC point estimate is not accompanied by uniformly better fold stability.

## P2: does learned window importance improve fixed averaging?

P2 preserves the two stages and96D cross-seizure summary.26shared zero-initialized scorer coefficients act on24latent coordinates,t/20 and(t/20)^2,with masked softmax inside each stage/channel. It has10403parameters(+26,+0.25% versusFull),not exactly the same capacity.

P2-minus-Full differences are Macro-F1 **+0.000213**,AUROC **+0.003094**,AP **+0.001628**,EZ-F1 **-0.000149**,MRR **-0.012313**,Top-1 **-0.025000**. The nominal paired intervals are[-0.004147,0.004492] for Macro-F1,[-0.001195,0.008805] for AUROC and[-0.003995,0.007499] for AP. These tiny point gains do not establish superiority over fixed averaging. No result supports a reliable overall localization gain from this particular attention scorer. This does not exclude other attention architectures; none were searched.

P2Macro-F1/AUROC fold SDs are0.032417/0.051593. That is a slightly smaller F1SD but a larger AUCSD than Full,not multi-seed robustness. All comparisons are conditional on this seed and historical model choices.

## A2: does the noEarly deficit merely reflect fewer parameters?

The unmatched noEarly has9329parameters versus Full10377. A2 retains its ordinary CNN and increases the single hidden head64to73,using the original64neuron slices and9effective additional neurons. It reaches10337parameters(-40,-0.39% versusFull),the nearest integer width in this ordinary one-hidden-head design. It has no inactive padding parameters and no FiLM/Late Gate.

Compared with unmatched noEarly,A2 changes Macro-F1 by **+0.006765**,AUROC **+0.008684**,AP **+0.009126**,EZ-F1 **+0.010629**. Nominal intervals are[-0.000456,0.014036] for Macro-F1 and[0.001166,0.018293] for AUROC. The AUC additional-family lower bound is-0.001296; the corresponding primary family does not establish a simultaneous Macro-F1/AUROC improvement.

Compared with Full,A2 has Macro-F1 **-0.002629**,AUROC **+0.009412**,AP **-0.015784**,EZ-F1 **-0.005453**,MRR **-0.043167**,Top-1 **-0.100000**. A2-minus-Full nominal intervals are[-0.014669,0.009497] for Macro-F1 and[-0.005871,0.027732] for AUROC. Full's Top-1 advantage has nominal interval[0.025,0.175],but this is an auxiliary unadjusted comparison,not a substitute for the predeclared main metrics.

Capacity matching narrows the F1 point gap and raises AUROC. Therefore the unmatched noEarly difference cannot be wholly attributed to unique physiological modulation. Full retains better AP/MRR/Top-1 point estimates,but EarlyFiLM is not established as indispensable for F1/AUROC. This controls approximate parameter count at the head,not exact functional capacity,compute or every possible ordinary CNN/head alternative.

## Uncertainty,source stability and interpretation limits

10000seed42paired resamples of80patients produce90nominal intervals (5absolute and5paired contrasts across9metrics). Ten additional Bonferroni-percentile intervals cover the five Macro-F1/AUROC pairs. No contrast has both family lower bounds above zero. There is no seedSD and no training-seed robustness claim. [All intervals](public/INTERVALS.csv),[family intervals](public/PRIMARY_FAMILY_INTERVALS.csv),[all fold directions](public/FOLD_DIRECTIONS.csv).

Effects vary by source. For example,P1 AUROC rises in HUP/LZU/multicenter but falls in pediatric patients (0.756699 versusFull0.793550). A2 Macro-F1 falls at HUP/LZU but rises at multicenter/pediatric. Aggregate changes cannot be described as uniform cross-source gains. Sources remain in the full cohort without quality-based exclusion. [Source table](public/SOURCE_RESULTS.csv).

Pooled-channel metrics are separate:Full pooled Macro-F1/AUROC0.664438/0.710978 are not the primary patient averages0.653465/0.765286. [Pooled](public/POOLED_RESULTS.csv),[folds](public/FOLD_RESULTS.csv),[fold SD](public/FOLD_SD.csv),[coverage](public/RAW_COVERAGE.csv).

An outcome-free engineering overlay removed invalid timestamp inputs before attention matmul after the finite-gradient gate caught0*NaN in padded positions. It leaves valid forward/derivative math and all13original scientific files unchanged.320label-free legal FIT/validation support checks passed before continuation; completed P1fold1 files/epochs were preserved. No failed P2epoch completed,no newOuter was viewed during repair,and no finite check was bypassed. [Audit](RUN_AUDIT.md),[repair proof](public_admission/ENGINEERING_REPAIR_ADMISSION.json).

## Direct answers

1. **Temporal partition:** not shown to be indispensable;P1 is competitive with mixed tradeoffs,but does not prove equivalence or isolate partition alone.
2. **Learned attention:** this P2 does not establish a meaningful advantage over fixed averaging.
3. **EarlyFiLM versus capacity:** approximately matched ordinary-head capacity recovers much of the F1 point gap and has higher AUC;the evidence does not justify attributing the unmatched deficit entirely to physiological modulation. Full's auxiliary ranking advantages remain reported.

No main-model replacement or further search follows from these already viewed Outer results. Negative results are retained. All data,identities,individual scores,weights,preprocessors,caches and runtime logs stay private.
