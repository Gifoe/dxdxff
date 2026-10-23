# Four-source LOCO, five fixed models: seeds42/43/44

Completed2026-10-23 with independent PASS:60 logical cells,20 historical seed42
reuses and40 fresh seeds43/44 fits;1200 patient/seed rows,10800 endpoints,405
nominal intervals and8 main-family intervals. All60 choices froze before NEW
test inference, all60 scores sealed before metrics. Actual new neural epochs424.

| Fixed model | Macro-F1 mean+/-SD | AUROC mean+/-SD | AP mean | EZ-F1 mean | MRR mean | Top-1 mean |
|---|---:|---:|---:|---:|---:|---:|
| Raw+Feature | 0.625048+/-0.007563 | 0.745515+/-0.010748 | 0.549025 | 0.430256 | 0.772382 | 0.675000 |
| Raw-only | 0.589337+/-0.012305 | 0.706683+/-0.003158 | 0.517056 | 0.384000 | 0.760578 | 0.679167 |
| Feature-only | 0.615259+/-0.003714 | 0.750889+/-0.003261 | 0.542931 | 0.419440 | 0.745588 | 0.637500 |
| Linear SVM Feature+PR | 0.610443+/-0.002275 | 0.738479+/-0.000000 | 0.537266 | 0.409868 | 0.728211 | 0.637500 |
| Original TimeConvCNN Raw/noPR | 0.446847+/-0.028188 | 0.586908+/-0.011922 | 0.376882 | 0.179052 | 0.492577 | 0.370833 |

See [FINDINGS.md](FINDINGS.md) for all9 metrics, allseeds, sources, paired
intervals and negative findings. Full-minus-own-Raw-only Macro-F1/AUROC remain
positive under8-test correction; stable superiority to Feature-only/SVM is NOT
established. Feature-only has the higher mean AUROC, Raw-only higher Top-1.

A reporting writer namespace bug was repaired OUTSIDE the12 frozen scientific
files: exact existing numeric tables copied, source43 public metadata restored,
original independent verifier rerun unchanged. No retraining/inference/threshold
change. See [reporting bridge](../../tools/drechan_loco_3seed_reporting_repair_v1/README.md)
for the executable report-stage entry and public_supplement for immutable-artifact
proof. Source, all34 anonymous package payloads and supplementary proof are released.

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
