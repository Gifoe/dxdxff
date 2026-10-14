# Macro-F1>0.70 investigation and frozen Outer — completed, target failed

**Scientific terminal: `MACRO70_TARGET_NOT_MET_AND_OUTER_CALIBRATION_FAILED`.** This execution did not achieve0.70. The best repeatedly fitted Development decision rule reaches0.695249; the frozen80-patient Outer reaches0.614959, below originalM8's0.634097. Do not present the validation calibration gain as model generalization, an independently confirmed improvement or a new clinical result.

## Primary Outer result

All estimates below are equal-weight patient means over80 unique own-fold held-out patients/7,635 canonical channels, not pooled-channel estimates or Development folds.

| Metric | Original D1 | Original M8 | M18 time control | Final frozen decoder |
|---|---:|---:|---:|---:|
| AUROC |0.732636|0.762397|0.742704|0.742704|
| AP |0.527175|0.554067|0.541925|0.541925|
| Macro-F1 |0.616556|0.634097|0.621556|0.614959|
| EZ-F1 |0.408531|0.434087|0.421900|0.420444|
| NEZ-F1 |0.824582|0.834107|0.821212|0.809474|
| Balanced accuracy |0.665741|0.681352|0.672722|0.670491|
| EZ sensitivity |0.477193|0.497392|0.503887|0.512637|
| NEZ specificity |0.854288|0.865313|0.841557|0.828345|
| MRR |0.689399|0.749310|0.762215|0.762215|
| Top1-EZ |0.575000|0.637500|0.675000|0.675000|
| Accuracy |0.748950|0.760166|0.746787|0.735265|
| Predicted EZ fraction |0.210810|0.209195|0.231007|0.239974|

Full precision, all five folds/centers, pooled confusions and all paired uncertainty are in `macro70_outer/results/`.

Final minus D1 Macro-F1 is **-0.001597**, paired95% CI **[-0.025047,0.022773]**. Final minus originalM8 is **-0.019138**, CI **[-0.038906,0.000361]**. Final minus its ownM18 global-threshold model is **-0.006597**, CI **[-0.023787,0.010015]**. None supports a positive Macro-F1 improvement. OriginalM8 has higher point Macro-F1 here; it is retained as the original comparator, not switched into the selected new recipe after viewingOuter.

The final decoder has exactly the same ranking asM18: AUROC/AP/MRR/Top1 do not change under a constant within-patient cutoff. Its AUROC is0.019693 below originalM8, CI **[-0.040819,-0.003809]**, under the exploratory unadjusted paired analysis. Adding source/dispersion threshold calibration cannot repair the underlying ranking/generalization loss.

## What the earlier0.66 meant

The previous0.666795 is **Development**:65 validation patient-fold appearances/47 unique IDs, selected using that same repeatedly used validation protocol. Its original frozenOuter was0.634097,80 different own-fold test evaluations over the same historical cohort. It was not an already achieved independentOuter0.66.

| Recipe / stage | Development patient Macro-F1 | Outer patient Macro-F1 |
|---|---:|---:|
| Frozen D1 |0.644546|0.616556|
| Original M8 |0.666795|0.634097|
| M18 raw time control |0.669983|0.621556|
| Final source/dispersion decoder on M18 |0.695249|0.614959|

FinalDevelopment-to-Outer difference is **0.080290**. The final fittedDevelopment point still misses0.70 by0.004751. That is not rounded up to a target success. The experiment establishes failure within these tested recipes, not a mathematical theorem that no future method can reach0.70.

## Actual search, including unsuccessful outcomes

There were **50 new supervised model/fold training cells**,393 positive neuraltraining epochs and2,000 actually fitted boosting trees across phases4–7. There were **495 validation-calibration cells** across phases8–14, corresponding to99 declared global policy configurations evaluated onfivefolds. These are calibration fits, **not495 freshly trained CNN models**, and some duplicated controls remain explicitly retained. All protocols were frozen before their corresponding phase execution; later phases were adaptive responses to priorDevelopment outcomes, not a single preregistered independent confirmatory study.

| Phase | Mechanism | Cells | Best fitted Development Macro-F1 |
|---|---|---:|---:|
|4|Patient-context shift/readout/temperature|15 training|0.668015|
|5|Joint morphology/physiology readout and residual cap|10 training|0.667791|
|6|Nonlinear conditional feature-complementarity trees|10 training|0.666807|
|7|Native spectral/time morphology, matched time control|15 training|0.669983|
|8|Validation-only source-specific cutoffs|30 calibration|0.692412|
|9|Fixed label-free patient score-location offset|40 calibration|0.693045|
|10|Source/offset rules on five earlier trained recipes|75 calibration|0.691674|
|11|Patient score-dispersion calibration|80 calibration|0.694013|
|12|Dispersion/inverse-dispersion on earlier recipes|150 calibration|0.691291|
|13|Fixed equal-logit branch combinations, not single modules|60 calibration|0.693092|
|14|Exact source-validation confusion plateaus vs0.005 grid|60 calibration|0.695249|

Complete candidate means are in `calibration_audit/results/ALL_NEW_CANDIDATES.csv`, including repeated comparators; eachphase's own `results/` retains its full mean/fold/center/calibration or training tables and actual independent verifier. No unfavorable model/fold was dropped, no best architecture was chosen independently per fold, and no patient/label-derived threshold oracle was reported as legal performance. The earlier diagnostic0.728583 used actualvalidation labels to select a different optimal cutoff per patient and is only an upper-bound diagnostic.

The multiple-branch combinations do not beat the best single-branch fitteddecoder; none is used in the finalOuter. They are explicitly marked as combinations rather than disguised as one small module. The spectral-only branch is weak; the combinedtime/frequency branch has betterDevelopment ranking but does not win primaryMacro-F1. Source calibration contributes the largest apparentDevelopment classification gain without changing ranking, exposing why fittedF1 cannot be equated with improved representation.

## Exactly what was selected, before Outer

The globalbest validated policy is `M18_TIME_CONTROL__EXACT_SOURCE_LOCATION_0.0_SCALE_0.5`. M18 is a frozenD1 plus small time-CNN physiology-guided correction, same-channel genuine pre-reference, early/late means, across-seizure mean/SD, patient-relative embedding and processedR14 entering a single correction head.10,377 correction parameters were trained; D1's1,537 parameters stayed frozen. Residual cap1.5 and all five selected weights were fixed beforeOuter.

The classifier's originalNEZ probability remainsunchanged forranking. A label-free populationSD of logit probabilities is computed within each patient's observed channels. Source-specific cutoffs fitted on originalvalidation groups apply to a square-root-dispersion normalized decision scale; the effective cutoff is back-transformed into the originalprobability scale. Offsetcoefficient is0, scaleexponent0.5. There is no inference patientlabel, trueEZ count or per-patient threshold oracle. Source identity is extra knownmetadata: this is **not a source-free decoder**. Tiny source-validation groups can fit highly optimistic cutoffs; their learned maps stayprivate.

The exact plateau-resolution check raises the fittedpoint only0.001235 over the0.005-grid best. Threshold grid resolution is not the main missing0.70 explanation. No Source/scale/threshold value is changed after viewing thisOuter.

## Why the new rule fails outside its fit set

On secondary pooled7,635-channel confusions, M18 has TP778/FP944/TN4,948/FN965. Finalcalibration has TP786/FP1,079/TN4,813/FN957: it finds8 additionalEZ channels but introduces135 extra false positives. PooledMacro-F1 falls0.643675→0.630557, whereas primarypatientMacro-F1 falls0.621556→0.614959. Both effects are reported at their correct distinct units.

PatientNEZ specificity drops0.841557→0.828345; NEZ-F1 drops0.821212→0.809474. Higher sensitivity is not a uniformly betteroperating point. Relative toM18, calibration improves LZU patientMacro-F1 but harms HUP/multicenter/pediatric averages. No center-specific significance is asserted; pediatric has only8patients. Unknown-source fallback is implemented but was not invoked on theseOuter channels, so the observed failure is not explained by unseen-source fallback alone.

The evidence is consistent with repeatedvalidation/cutoff fitting optimism and mismatchedpatient score dispersion/operating points. It does not identify a specific biologicalcause or establish a label-noise ceiling. The newerM18 also loses ranking versusM8 onOuter despite slightly higherDevelopmentMacro-F1; decoder refinements cannot compensate for that. The tested complexity is not justified by the achievedgeneralization results.

## Fidelity, verification and privacy

The original80patient/256seizure/7,635channel cohort, fivefoldroles, bank identities, originalR14 imputer/scaler andFITphysiology scales remain unchanged. NoRaw/EDF or physiological feature re-extraction, preprocessing refit, additional seed, originalcheckpoint modification or installedskill occurred. The new onlineFFTbranch uses only physically available10–125Hz on genuine500-sample250Hzwindows, not reconstructedδ/θ or full80–150Hz.

Before the finalOuter, all15 selected modelcheckpoints, fiveD1 bankpreprocessors/FITscales, five sourcecutoffmaps, originalsplit/input/source hashes and the globally selected recipe were locked. Fifteen model/fold scores and five decision arrays were sealed before this execution's fullTESTlabel read. D1/M8 predictions were reused exactly from the previous verified frozenOuter seal; M18 was actually inferred afresh on all80own-fold patients. Reuse is disclosed, not counted as newly retrained controls. EmbeddedD1logit replay difference iszero in allfivefolds.

Independent audits cover all new training cells, every namedcalibration cell, all selected factories/state hashes, original own-foldTEST exclusion, source-map recomputation fromvalidation only, label-free offset/dispersion mathematics, per-patient sklearn metrics and10000seed42pairedID-cluster bootstrap via independentmultiplicity counts. The finalOuter verifier checks320patient metric cells, pooledconfusions, allweights/maps/source bindings and exactranking equality toM18. Downloadedaggregate hashes agree withtrusted-server verification records.

An initial newOuter prefreeze failed becauseWindows defaultGBK decoding could not read aUTF-8protocol string. This happened before a finalfreeze/testinference/label read; explicitUTF-8was added, the originalfailedsource retained privately, and scienceconfiguration unchanged. One successful final frozenOuter pass then completed. Native/transfer issues in previous prefreeze setup did not discard measured epochs or cause test-driven choices.

AllOuter is **`USER_AUTHORIZED_POSTHOC_OUTER_EXPLORATORY_REPEATED_TEST`**. This cohort has historicaloutcome exposure, and globalarchitecture/calibration selection uses validationappearances that can overlap other folds' testIDs. Ownfoldfitting excludes its TESTpatients, but this is not independently nestedarchitecture selection, an untouchedholdout or clinicalconfirmation. The uncertainty is unadjusted and does not remove adaptive model-selection optimism.

Code/hashes/anonymousaggregates are public. Rawsignals, identities, labels, per-channel predictions, weights, scaler/sourcecutoffmaps andlogs remain on thetrustedserver. Unrelated processes/data were notmodified. This boundedexecution is concluded with the abovefailure, not followed by an automaticOuter-driven rescue.
