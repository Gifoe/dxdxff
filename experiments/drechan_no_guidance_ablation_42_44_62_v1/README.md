# No-guidance DREChannNet: selected seeds42/44/62, complete ablations

The user fixed the existing no-Early/no-Late six-feature single-head graph as main and requested seeds42,44,62 AFTER seeing historical Outer results. Variance and Head Interaction remain absent. This is an **outcome-informed selected-seed exploratory repeated-test study**, not random-seed robustness or independent confirmation. Report all five previously viewed main seeds in a separate disclosure table; never hide43/52 or merge them into the primary estimate.

## Fixed matrix

| Group | Operation relative to main |
| --- | --- |
| FULL | Existing no-Early/no-Late main,9329parameters |
| EARLY_FILM_RESTORED | Restore exact old no-Late EarlyFiLM,10377parameters; labelled add-back reference |
| WO_FEATURE_PR | Remove direct physiological summary patientZ only |
| WO_RAW_PR | Remove finalRaw96D patientZ only |
| WO_BOTH_PR | Remove both; retain waveform normalization and own pre-reference |
| WO_PRE_REFERENCE | Remove own pre-ictal latent subtraction |
| RAW_ONLY | Remove all explicit physiology; keep RawPR/reference/original pooling |
| PHYSIOLOGY_ONLY | Corrected FeaturePR summary only |
| Six feature deletions | beta,lowgamma,partialhighgamma80-125,RMS,LineLength,SpectralEntropy10-125 individually; mask before relative statistics |
| P1_TEMPORAL_STATS | Whole0-20meanSD then cross-seizure meanSD96D, no guidance |
| P2_TIME_ATTENTION | Masked time-aware weighting within original stages,96D,no guidance;26extra coefficients |
| CAPACITY_MATCHED | NoEarly ordinary head64to73,10337parameters,approximately matched to restoredEarly |

17groups x3seeds x5folds=255logical cells. Exactly50strict reuses (main15,Raw/Phys42/44twenty,restoredEarly42/44ten,capacity42five) plus205fresh fits. No old withEarly PR/feature-deletion or pooling substitute. All fresh models consume full original **untrained** initializer before deletion; no trained-main initialization.

The already absent Early/Late/BothGuidance/windowPR deletions are aliases of FULL, not duplicate fits/hypotheses. FiLM mismatch and Early/Late factorial interaction are inapplicable. Only the PR factorial interaction is identifiable.

Same original80patients,7635channels,256seizures,1471965validRaw250windows and patient-isolated fivefold roles; original FP32/TF32off/chunk4096/full-patient updates/loss/optimizer/budgets/validation-only epoch and threshold rules. No new data, features, extraction, performance gate, hyperparameter search or test tuning.

All255selections freeze before newOuter; all255scores seal before metrics. Report9patient-equal metrics,eachseed/metric mean/sampleSD,pooled separately,fold/sampleSD/source/coverage,306nominal and32additionalfamily intervals from10000seed42paired draws of80unique patients AFTER averaging seed metrics. Do not average predictions or select the best seed. Negative findings retained. See PROTOCOL_LOCK.json.

Server code/weights/runtime onD, cache read in place. Sustained foreground authenticated owner required; diagnose failures and preserve checkpoints/Adam/RNG/logs before exact resume with new RunTag. Source and admission evidence must be pushed and remote-verified before formal fitting.

Only source/hash/anonymous aggregate tables and reports are public. No waveforms,identity,individual scores/embeddings,weights,preprocessors,caches,logs or secrets. Actual experiment outputs will be added only after verified completion.
