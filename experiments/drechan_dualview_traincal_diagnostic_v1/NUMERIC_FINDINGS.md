# Existing FeaturePR input amplification:actual TRAIN/CAL evidence

## Verified arithmetic,not new model performance

The frozen diagnostic found large CALlogits in all3bodies. The follow-up used no labels,newRaw forward,model fit orclassification metric. It exactly reconstructed both original input banks in all20TRAIN/CALcases andchecked280coordinate summaries/60head-path summaries. All120independent float64 linear-projection comparisons passed. Original models,thresholds,preprocessors andOuterresults were not changed.

Coordinates below are numerical indices,not guessed biological labels. Indices8/9remain the previously source-verified inactiveVariance pair. Every coordinate,patient andwindow is retained. Full details are in[coordinate numerics](results/feature_numeric_extension/FEATURE_COORDINATE_NUMERICS.csv),[head-path scales](results/feature_numeric_extension/HEAD_INPUT_PATH_SCALE.csv)and[verification](results/feature_numeric_extension/INPUT_NUMERIC_VERIFICATION.json).

| CAL fold | Active coordinate | Maximum absolute value before frozen global scaling,FeaturePR | Frozen TRAINscale | Maximum absolute value after scaling |
| --- | --- | ---: | ---: | ---: |
|3|coord_02|10.04358963|2.48082194e-9|4.048492800e9|
|5|coord_02|6.48259809|2.16479752e-9|2.994551808e9|
|5|coord_04|6.87348016|4.41910866e-10|1.555399680e10|
|5|coord_05|7.30501930|4.91332717e-10|1.486776525e10|

This is amplification by the existing TRAIN-only standardizer,not a model-generated feature. It is not evidence that the underlying physiological property is biologically useless. The same noFeaturePR global preprocessing does not show comparable billion-scale values:its largest active CALmagnitude is160.516083 in this audit. No control is modified to create this difference.

## Entry into the unchanged classifier

For FULL,meanCALraw preactivation RMS across3seeds is0.689893in fold3and0.597646in fold5. The physiology preactivation RMS is1.750556e7and1.303944e8,respectively. For PRUNED_REF the corresponding rawRMSvalues are0.477784and0.537971,versusphysiology1.720299e7and1.311499e8. Other folds' physiology/rawRMSratios are of order1,not millions.

These are float64 contributions to the first linear layer before jointGELU,not additive decompositions of the final nonlinear logit. They establish that extreme standardized physiology enters anddominates that preactivation. They do not isolate the cause of the observed Macro-F1 loss orshow that dropping physiology orrefitting thresholds helps. All weights are the original frozen weights.

## Unit-consistency issue to test next

The inherited`patient_z`computes`(x-patient_mean)/patient_sd`whenraw-unit SD>1e-8,butotherwise divides by1. Thus one branch is dimensionless whilethe fallback retains centered original units. A later TRAIN-onlyStandardScaler can learn a tiny scale from fallback-unit training values,andthen greatly amplify a CALpatient that crosses the absolute1e-8boundary anduses the dimensionless branch. The observed scales andvalues are consistent with this mixed-unit mechanism;no patient identity orlabel is needed to produce the arithmetic failure.

The next justified experiment is a separately locked,unit-consistent patient-relative preprocessing comparison,not another large unmotivated architecture search. It should use only TRAIN-derived reference units,handle truly degenerate contrast coordinates explicitly,andkeep all6features/allpatients/windows. Keep a meaningful TRAIN-only noFeaturePR control,not a weakened replacement. Verify clean versus legacy equality where both use genuine standardization,andseparately test unit-rescaling invariance andconstant-coordinate behavior. Then retrain matched full/PR/core controls under the original budgets before any new assessment.

This is a next-test rationale,not a deployed correction orpromise of better results. InitialRawanchor drift also exists,so the numeric finding does not explain every failure. Prior negative calibration experiments remain negative. The original80-patientOuterMacro-F1=0.653828remains unchanged;the active performance/contribution goal is still unmet.
