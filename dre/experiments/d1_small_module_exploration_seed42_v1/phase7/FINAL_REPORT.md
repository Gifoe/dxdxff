# Native morphology — executed, target not met

| Development recipe | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| D1 |0.725978|0.503601|0.644546|0.443537|
| M16 spectral |0.728729|0.520399|0.652384|0.455576|
| M17 time+spectral |0.762931|0.542687|0.661649|0.481037|
| M18 time control |0.753702|0.543765|0.669983|0.486255|

All15foldcells/148positive trainingepochs completed; allfour independentFFT/coordinate/identity/mask/permutation/gradient tests passed locally/server beforefreeze. ActualFITgradient admissions were discarded beforemeasuredtraining. All15selected factory checkpoints were freshly replayed, originalD1weights unchanged, and195patient metrics/bootstrap/scaler/roles audited. BestMacro-F1isbelow0.70; the spectral-onlybranch is weak andtime+spectralrankingdoesnotconverttohighestF1. No test was read duringthisphase. Subsequent boundedcalibration/frozenOuter is in the root `MACRO70_FINAL_REPORT.md`; no physiologicalcausality or clinicalconfirmation is inferred.
