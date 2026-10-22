# Four-source LOCO, five fixed models: seeds42/43/44

Prospective extension of the completed seed42 protocol, not a new architecture or search.
Hold the ORIGINAL seed42 FIT/validation/test patient roles fixed. Vary neural
initialization, training patient order/dropout, and SVM patient-group OOF calibration
with train_seed+1009*held_index. All hyperparameters/threshold grids, six features,
PR definitions and complete-patient windows remain unchanged. Fresh seeds43/44
are fixed BEFORE fitting and test viewing; do not drop any source or unfavorable seed.

60 logical cells = 5models x 4held sources x 3seeds. Strictly reuse20 old seed42
choices and sealed predictions; train40 fresh main cells plus24 auxiliary OOF SVMs.
The old20 were already reported; do not pretend all60 tests are newly blinded.
All60 choices freeze before any NEW test; all60 score cells seal before metrics.
No old fivefold weights. Current six-feature main has no Variance, Early FiLM,
Late Gate, Head Interaction. All80patients/7635channels/256seizures/1471965valid
Raw250 windows, complete-patient updates, original FP32/BF16 regimes and budgets.
No new extraction, data, features, performance gate, sampling or tuning.

Primary: average each patient's endpoint across3seeds, then equal80patients.
Report each seed, mean/sampleSD, per-source, four-source equal, sourceSD, pooled
PER SEED, all9 endpoints,405 nominal intervals and8 corrected main contrasts.
10000 seed42 paired patient draws WITHIN source after metric seed averaging.
Never average probabilities as the main endpoint and never choose the best seed.
Independent verification requires1200patient/seed rows and10800endpoint values.

Code/runtime/weights stay server D-drive; read existing caches in place. Formal
source and real admission proof must be pushed and remote-verified BEFORE fit.
run_all.ps1 has exclusive own-job locking and atomic epoch checkpoints. Preserve
optimizer/RNG/logs, diagnose genuine engineering failures before exact resume.
All data/IDs/private scores/checkpoints/logs remain trusted-server-only. Only
explicit anonymous aggregate allowlists can be exported.

Source labels are not proven individual hospitals. TimeConv's time context,
pooling, precision and objective differ: compare full pipelines, not one causal
architecture factor. This repeatedly used historical cohort remains exploratory.
Three fixed training seeds do not constitute independent external confirmation
or uncertainty over unseen hospitals. No installed research/extraction skills.
