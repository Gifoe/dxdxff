# Existing-recipe source/location calibration

Fivepreviouslytrained/frozenbases×threeglobalcoefficients×fivefolds:75actual calibrationcells. No newneuraltraining, preprocessing/dataextraction ortestread. Exactvalidationpredictions/checkpoints andoriginalownfoldroles arechecked; inferenceoffsetreceivesnolabel. Onecommonbase/coefficient ischosen, notbestarchitectureperfold.

BestfittedDevelopmentMacro-F1: **0.691673839632**, `M9_CONTEXT_SHIFT__SOURCE_OFFSET_0.0`, below0.70 andnotaboveearlierphasebest. Independentverifier recomputes75maps/975patient metrics and10000clusterbootstrap. All15configs/pooroutcomes are retained. Thephaseisadaptive,notanindependentconfirmation; rootfinalreport supplieslaterOuter.
