# D2 Source-Rank Compression seed42

Frozen R4/D2 versus scratch-trained source bias R0, rank1 R1, rank2 R2.
Completed on the original Windows server: all 15 new fold-variant runs,
169 epochs, 34 pretraining engineering checks and 15 FIT-only smokes.
Terminal: `RANK4_REMAINS_BEST`.

| Model | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC |
|---|---:|---:|---:|---:|
| R4 (frozen D2) | 0.647926 | 0.443199 | 0.519936 | 0.718221 |
| R0 (source bias only) | 0.636982 | 0.424982 | 0.517063 | 0.707296 |
| R1 | 0.643679 | 0.435388 | 0.518756 | 0.711650 |
| R2 (registered primary) | 0.644188 | 0.439254 | 0.514952 | 0.717052 |

R2-R4 Macro-F1 = -0.003738, paired 95% percentile CI
[-0.009900, +0.002256], positive in 1/5 folds. No compressed arm
passes the continuation gate; none reaches 0.700. This rejects the
registered gain criterion, not a claim of statistically proven R4 superiority.
See FINAL_REPORT.md for all 11 metrics, six paired contrasts, source effects,
mechanisms, interventions and explicit answers to the 24 requested questions.

Source-direction interpretation: across the 30 source-pair/fold cosines,
16 are negative for each of R1, R2 and R4. R1 ranges from -1 to +1
(rounding at machine precision); R2 from -0.999996 to +1.000000;
R4 from -0.999977 to +0.999975. Thus directions include both aligned and
opposing pairs, not uniformly aligned corrections. These extrema do not
establish that higher rank is redundant: effective numerical ranks are
exactly 1/2/4 in every corresponding fold, and the compressed models
show no registered generalization gain. Source-specific collapse beyond
the prelocked 0.050 Macro-F1 gate was absent, but Pediatric declines in
all compressed arms. Removing R2 direction/bias alone changes Macro-F1
by -0.009201/-0.006066; both components contribute to this trained scorer.
Neither observation proves that rank4 overfitting is the original bottleneck.
Exactly15 formal runs; primary hypothesis R2-R4. Same original 88D and five
patient-held-out development folds; no outer TEST, rank3 or rescue search.

Read SOURCE_AUDIT.md and PROTOCOL_LOCK.json before running. Pipeline:
code/prepare.py -> tests/engineering.py -> code/run_training.py ->
code/finalize.py -> tests/completion_verify.py. Each server entrypoint takes
`--protocol <absolute PROTOCOL_LOCK.json path>`. Finalization is one-shot;
inspect existing status/ledger before recovery instead of silently rerunning.
Original private FIT/VAL banks and hashes are required; no raw EEG extraction.
Only source and aggregate audits/results are released. This is repeatedly
used development evaluation, not independent clinical or unseen-center testing.
The 101 original private artifacts remain hash-identical after execution.
Frozen training source matches the pretraining engineering audit. Aggregate
downloads are SHA-checked by tests/public_delivery.py before publication.
