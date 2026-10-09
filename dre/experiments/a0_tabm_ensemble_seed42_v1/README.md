# A0 + TabM-style parameter-efficient ensemble, seed42

Requested controlled experiment: frozen original A0 (8,817 parameters), ordinary
width111 W1 (10,167) and K4 width96 shared-parameter W2 (10,132). Only ten new
W1/W2 five-fold development runs; no A0 retrain, outer evaluation or search.
Use the existing branch; no additional branch is created.

Scientific scope and sources: [SOURCE_AUDIT.md](SOURCE_AUDIT.md),
[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json), [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md).
Completed on the trusted server: **TABM_NOT_SUPPORTED**, development gate failed.
Patient Macro-F1: A0 0.638080, W1 0.636871, W2 0.637767. W2−A0
−0.000313, paired 95% interval [−0.009571,+0.008879]. No outer TEST or
follow-up training. See [FINAL_REPORT.md](FINAL_REPORT.md), all eleven
[aggregate metrics](results/VALIDATION_SUMMARY.csv), and
[paired intervals](results/PAIRED_BOOTSTRAP.csv).

Trusted private server execution order:

1. `code/prepare.py`: verify original feature/split/checkpoint hashes, exact A0
   preprocessing/prediction replay; seal FIT/VAL-only original88D arrays.
2. `code/smoke.py`: upstream AST/initial/output/gradient parity, six synthetic
   groups, discarded real FIT-only smoke.
3. `code/run.py`: all five scratch W1 folds followed by all five scratch W2 folds.
4. `code/finalize.py`: frozen selected predictions, bootstrap, member/center/
   error/threshold/ranking/efficiency audits and development gate.

All commands expose `--help`. Private input/checkpoint paths are required, not
bundled. `base.py` imports the SHA-locked sibling PR-UAS source. Upstream parity
uses `smoke.py --official-source` pointing to exact official tabm.py at the pinned commit, loaded
without unrelated embedding imports. Isolated components are Apache-2.0; see
[LICENSE_TABM.txt](LICENSE_TABM.txt). No full published TabM reproduction is claimed.
Git line-ending conversion is disabled here to preserve recorded source hashes.

Public files are source and aggregate findings only. Features, labels/IDs, individual
predictions, checkpoint/optimizer/RNG states and runtime logs stay on the server.
The65 validation appearances from47 IDs are development-selected, not independent
test performance. No patient/member-specific deployed threshold or model selection.

Required JSON audits and final status live in `audit/`; compact CSVs live in
`results/`. `tests/test_tabm.py` contains the unchanged pretraining six-group
admission suite; `tests/test_audit_tables.py` separately covers a post-training
metadata-table engineering repair. `finalize.py` requires the private completed
training status and is an aggregation step, not a training or inference replay.
