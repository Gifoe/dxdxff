# Stable conditional feature pruning, seed42

Source-free compact F0: ABS+DELTA+RATIO, no ZDELTA, original Patient Z,
original five patient-held-out folds. This study does **not** run a new outer
TEST. Reused Development is exploratory, not independent confirmation.

The executable protocol is `PROTOCOL_LOCK.json`. `code/run.py` imports and
hash-verifies the authoritative original trainer, banks, initialization,
feature manifest, split, F0 checkpoints and frozen validation predictions.

Stages: algebraic LN88 folding and parity; 330 nested FIT-only screening cells;
freeze per-fold subsets; matched Development B1/B2/B3/B4; fixed-score paired
10,000-draw patient-ID bootstrap and independent metric verification.
No eligible groups means B3/B4 alias B1, not extra nominal training results.

```powershell
python -m pytest PATH_TO_EXPERIMENT/tests -q
python PATH_TO_EXPERIMENT/code/run.py --runtime PRIVATE_NEW_RUNTIME --source TRUSTED_SOURCE_ROOT
```

Existing runtime resumes only hash-bound completed tasks. Do not change source
or protocol to resume. The required historical dependencies must remain
available; there is no fallback to a guessed manifest or a similar model.
The trusted run uses the previously validated Python3.13 environment, importing
NumPy before Torch. Installed skills are not used.

`code/verify_screening.py` performs the extra post-run score-only audit. It was
added after competitive execution and verifies that run's original source hashes.
The completed original runtime intentionally refuses a new training binding
that includes an added file; fresh reproduction uses a new runtime. Do not
rewrite the original freeze to accommodate later audit scripts.

The local `.gitattributes` preserves exact artifact bytes, including server CSV
line endings, so Git checkout does not invalidate their recorded hashes.

`FEATURE_IMPORTANCE_BY_INNER_FOLD.csv` gives deletion-minus-full effects on
untouched FIT screening patients, not Development or outer results. Final
Development comparisons separately include optimization effects of folding.
Parameter and column reductions do not imply clinical irrelevance.

Only source, hashes and non-identifying aggregates are public. Runtime banks,
patient/channel IDs, labels, scores, checkpoints, preprocessing state and logs
remain private. See `FINAL_REPORT.md` for actual completed results and limits.
