# Validation-only source cutoff calibration

Two alreadyfrozenbase choices×three sourcepolicies×fivefolds:30actual calibrationcells, noCNNtraining. Protocol isfixedbeforeexecution; the originalvalidation-only0.005thresholdgrid and ownfoldFIT-sourcecoverage gate are retained. Everyactualsource map remainsprivate; unknownsources use theoriginalglobalthreshold. Tinyvalidationgroupsare explicitly counted and are notindependentcalibrationevidence. Syntheticmapping/fallback/shrinkagetests passed beforefreeze.

BestfittedDevelopmentMacro-F1: **0.692411614000**, `M18_TIME_CONTROL__SOURCE_MIN1`, stillbelow0.70. Scores/ranking areexactlyunchanged; gainisdecisioncalibration, nota better encoder. Independentverification recomputes all30maps/390patient metrics andpaired10000clusterbootstrap. Allsixglobalconfigurations remain in `results/`; noOuter is used tofit/selectthem. Rootfinalreport documents thelateractualfailedOuter.
