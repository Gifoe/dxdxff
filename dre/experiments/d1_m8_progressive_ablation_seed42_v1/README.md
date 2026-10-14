# D1 → M8 progressive ablation, seed42

Fixed matrix: A0 Raw ABS14/no PR, A1 Raw DELTA14/no PR, A2 original frozen D1, A3 original frozen M8. Only A0/A1 are newly trained (ten cells). Historical engineered ABS18, 88D no-PR models, M18 and later calibrators are not substitutes.

Execution order on the trusted server: `code/prepare_features.py`, `code/train.py`, `code/evaluate.py`, `code/bootstrap.py`, `code/verify.py`, each with the same `--runtime` argument and pinned original dependencies. Feature preparation rejects any DELTA14 mismatch before training. Evaluation freezes selections/preprocessors, performs the ten new inference cells and reuses original D1/M8 score ledgers, then seals scores before the new full Outer label read. An existing matching completed runtime is reused rather than run as another scientific test.

The exact contracts and byte hashes are in `PROTOCOL_LOCK.json`. Private source paths are runtime defaults in the executable code, not patient-record listings. No private input, identity, clinical label, model weight, scaler, individual prediction or runtime log is published.

Primary Outer estimates weight the same80 unique held-out patients equally. Development has65 patient-fold appearances/47 unique patients and is reported separately. Both cohorts were extensively reused historically; this is exploratory repeated-test evidence, not blind or independent clinical confirmation. The M8 increment includes its physiology gate and changed objective, not a pure CNN-only effect. A0<A1<A2<A3 is tested, not imposed.

Read `FINAL_REPORT.md` for actual completion, metrics and progressive contrasts. No installed skill is used.
