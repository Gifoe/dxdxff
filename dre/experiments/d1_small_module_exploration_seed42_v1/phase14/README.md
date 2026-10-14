# Final exact validation operating-point resolution check

Twobases×twolocationcoefficients×threeexponents×fivefolds:60actual calibrationcells. Thischangesonlyvalidation-sourcecutoffresolution: midpointsbetweeneveryadjacentuniqueprobability plus0/0.5/1representallconfusionplateaus. Sourcecoverage, no-labelpatientSD/median, originalranking, defaultunknownsourcecutoff andtieordering arefixed. Brute-forceindependentplateautests passed locally/server beforefreeze. No testthreshold isoptimized.

BestfittedDevelopmentMacro-F1: **0.695248584133**, `M18_TIME_CONTROL__EXACT_SOURCE_LOCATION_0.0_SCALE_0.5`, below0.70. Independentaudit recomputes60maps/780patientmetrics and10000pairedbootstrap; globalcomparisonagainstallcompletedphasesconfirmsthisrecipebeforeOuter. Gainoverprior0.005-gridbestis0.001235,notatargetsuccess.

TheuserrequestedOuterwasthensafelyexecutedwiththisrecipefrozen,evenwithnegative0.70feasibilityfinding. Itsactual80patientMacro-F1is **0.614959**, beloworiginalM8 **0.634097**. Neitherthesourcemaps nor model/threshold/normalizationrule waschangedaftertest. Thisboundedexecutionisclosedwithnegativeevidence,notanautomaticrescuesearch. Read root `MACRO70_FINAL_REPORT.md` and `macro70_outer/results/`.
