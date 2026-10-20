# Complete six-feature F0 ablations: three training seeds

Implementation in progress; no new three-seed outcomes are claimed. The fixed main is the completed relative-guidance Full, not an ablation chosen from Outer.

The approved matrix contains 19 groups, seeds42/43/44 and the original five patient-held-out folds:285 logical cells and80patients per group/seed. Current seed42's45cells are reused after exact checks. Five historical seed42 Raw-only cells are optional strict reuses; all other missing cells are freshly fitted,235–240new fits. Different-architecture33-patient internal studies are not substituted.

Keep the original six measurements (beta,low gamma,partial high gamma80–125Hz,RMS,line length,entropy10–125Hz), original80patients/7635channels/256seizures/1471965validRaw windows, preprocessing definitions, matched initializers, losses and budgets. Variance and head interaction remain absent. All groups finish Outer without a performance eligibility gate.

See `PROTOCOL_LOCK.json` for the locked matrix/operations/statistics/privacy. Source is pushed before formal training. All285selections freeze before unfinished Outer inference; all285score cells seal before unified metrics. Primary results are averages of patient metrics across seeds, not a probability ensemble or best-seed selection. Independent verification covers4560patient-seed rows/41040endpoints/351nominal/36additional-family intervals.

Only code,hashes and anonymous aggregates are published. Signals,identities,individual predictions,weights,preprocessors,caches andruntime logs stay on the trusted server. These historical Outer results are exploratory repeated-test,not independent confirmation. Favorable ablation directions are not guaranteed or used to select a replacement main model.
