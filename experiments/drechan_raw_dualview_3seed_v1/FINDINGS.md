# Actual findings: global-plus-relative Raw views

## Decision

The fixed candidate is **not promoted**. It improved internal ranking but lost too much classification performance. Performance,expanded-reference and global-view qualification conditions failed. The complete[fixed gate](results/QUALIFICATION_GATE.json)is retained without edits. No newOuter was run and no better-looking control was substituted as the candidate.

These are33unique within-original-FIT ASSESS patients,three training seeds and40fold appearances per seed. They are not the original80-patientOuter. The historical main-model Outer Macro-F1=0.6538278218104221 andAUROC=0.7522551973793955 remain unchanged;subtracting the internal results below from those historical numbers is invalid. The correct internal reference isB0in the same table.

## Complete internal results

All values are equal-patient means after averaging repeated fold appearances within patient/seed and then the3seed metrics. These are not channel-pooled results or a probability ensemble. All90fresh fits/1091epochs and30exact reference reuses completed. See[all nine endpoints](results/MAIN_RESULTS.csv),[seed results](results/SEED_RESULTS.csv),[fold results](results/FOLD_RESULTS.csv),[source results](results/SOURCE_RESULTS.csv)and[separate pooled results](results/POOLED_RESULTS.csv).

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0:original architecture reference |0.619955|0.752344|0.572237|0.408406|0.831505|0.752759|0.820934|0.729798|-0.107229|
| PRUNED_REF:pruned-joint reference |0.625490|0.753058|0.564077|0.421698|0.829283|0.753704|0.819592|0.727273|-0.092279|
| FULL:fixed dual-view candidate |0.600308|0.762056|0.585878|0.379927|0.820689|0.749149|0.873770|0.815657|-0.102311|
| WithoutFeaturePR |0.560582|0.722467|0.545433|0.308427|0.812737|0.730715|0.774696|0.664141|-0.120805|
| WithoutRawPR |0.564538|0.755792|0.570848|0.345579|0.783497|0.717222|0.851064|0.762626|-0.058138|
| WithoutbothPR |0.488972|0.716051|0.530192|0.234180|0.743763|0.679218|0.780759|0.659091|-0.046212|
| Withoutglobalview |0.618986|0.743937|0.550959|0.410410|0.827562|0.750492|0.821254|0.729798|-0.093726|
| Withoutpre-reference |0.586042|0.696265|0.520811|0.346313|0.825771|0.742650|0.793091|0.702020|-0.127098|

FULL-minus-B0:AUROC+0.009712,AP+0.013641,Macro-F1-0.019647,EZ-F1-0.028479. All3seeds improved AUROC andAP but reduced Macro-F1 andEZ-F1. The Macro-F1/EZ-F1 losses exceed the predeclared0.01margin. The observed AUROC improvement also has a nominal paired interval crossing0,so it is a directional finding,not a statistically resolved improvement.

## Which contributions are supported?

Signed differences below areFULL minus the indicated retrained ablation. Each of the6new arms has15473parameters,matching initialization slices,patient-isolated roles and training budgets. Missing-context controls retain meaningful dimension-matched within-channel alternatives,not zero branches. Reference models have11425or9329parameters,so improvements over references alone would not isolate context from added capacity.

| Contrast | AUROC difference [nominal95% interval] | Macro-F1 difference [nominal95% interval] | Seeds with both positive |
| --- | --- | --- | ---: |
| FULL-minus-B0 |+0.009712 [-0.014554,0.035372]|-0.019647 [-0.042097,0.000521]|0/3|
| FULL-minus-PRUNED_REF |+0.008998 [-0.002648,0.020255]|-0.025182 [-0.043091,-0.008532]|0/3|
| FeaturePR |+0.039589 [0.018626,0.066238]|+0.039726 [0.022730,0.058051]|3/3|
| RawPR |+0.006264 [-0.013241,0.023567]|+0.035770 [0.019546,0.053034]|3/3|
| Globalview |+0.018119 [0.004214,0.031980]|-0.018678 [-0.038190,-0.000043]|0/3|
| Pre-reference |+0.065791 [0.032515,0.102779]|+0.014266 [-0.010018,0.037362]|2/3|

-FeaturePR has the clearest two-endpoint evidence in this candidate:both predeclared additional-family lower bounds remain positive(AUROC[0.012096,0.081712],Macro-F1[0.016340,0.067165]). This supports its conditional contribution in the tested architecture,not a universal physiological mechanism.
-RawPR improved both endpoint points in all3seeds. Its Macro-F1 additional-family interval remains positive[0.012069,0.061500],but the AUROC interval[-0.022858,0.030086]crosses0. RawPR therefore has not achieved the requested stable two-endpoint uncertainty evidence. One seed's AUROC gain was only0.000134.
-The global Raw view improved AUROC/AP/ranking while lowering Macro-F1 in all3seeds. Its additional-family intervals cross0for both endpoints. This is not an all-metric core-module benefit. The mean Macro-F1 is also lower than the pruned-joint reference,with an additional-family interval[-0.052383,-0.002297]forFULL-minus-reference.
-Pre-reference has a positive additional-family AUROC interval[0.017951,0.119821],but its Macro-F1 interval[-0.020603,0.046916]crosses0and one seed's Macro-F1 difference is negative. Do not label it a resolved two-metric contribution.

See[all144nominal intervals](results/INTERVALS.csv),[12predeclared additional-family intervals](results/PRIMARY_FAMILY_INTERVALS.csv)and[seed-direction evidence](results/COMPONENT_EVIDENCE.csv). The10000draw seed42 bootstrap resamples33unique patients while retaining their repeated paths/seeds. It is conditional on trained models and historical adaptive research;it does not include complete training randomness or correct the entire history of attempted architectures. Seed-direction consistency is not a substitute for uncertainty intervals.

## What the experiment changes,and what it does not

The global-plus-relative view corrected the previously negative RawPR AUROC point direction in this internal comparison,but did not solve overall Macro-F1 performance. BetterMRR/Top-1 does not imply that a fixed CAL-derived binary decision threshold transfers well to all held-out patients. This is a possible ranking/decision distinction,not proof that threshold selection caused the decline. The experiment did not identify a causal diagnosis of that decline.

No assessment-derived threshold,patient selection,post-hoc fusion weight,best seed or better-looking control is used to repair the result. Further investigation must use legal TRAIN/CAL data under a new prospective protocol,with matched controls and all prior negative results retained. Repeated assessment evidence must continue to be labeled exploratory. Final original-protocol80-patientMacro-F1>0.64,improvedAUROC and stable retained-module contributions remain unproven.

The private independent verifier passed960patient-fold-seed rows/8640metric values,144nominal/12additional intervals and actual role/window coverage,max interval discrepancy3.33e-16. The separate public checker verifies source/payload hashes and aggregate identities without private predictions. Public artifacts contain no individual predictions,identities,weights,anchors,preprocessors,raw/cache orruntime logs.
