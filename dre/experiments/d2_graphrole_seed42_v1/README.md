# D2-GraphRole seed42

Implementation-and-execution experiment: frozen D2 versus real GraphRole16 (G1),
within-patient shuffled GraphRole16 (G2), and neutral zero GraphRole16 (G3).
Exactly fifteen new formal runs on the five frozen development splits.
No outer TEST, model search, extra seeds, full GNN or rescue variants.

Completed terminal: `GRAPHROLE_SHUFFLE_CONTROL_NOT_BEATEN`.

| Model | Patient Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC |
|---|---:|---:|---:|---:|
| D2 (frozen D0) | 0.647926 | 0.443199 | 0.519936 | 0.718221 |
| G1 real graph | 0.649188 | 0.451017 | 0.531924 | 0.747376 |
| G2 shuffled graph | 0.653198 | 0.442994 | 0.532745 | 0.734149 |
| G3 neutral graph | 0.645147 | 0.443346 | 0.506121 | 0.719322 |

G1-D2 Macro-F1: +0.001262, 95% paired ID-cluster interval
[-0.015543,+0.020438]; G1-G2: -0.004009 [-0.019826,+0.013470].
Both comparisons improve in only 2/5 folds. The full continuation gate fails.
Graph dependence is measurable, but useful channel correspondence is not established.
Stop this registered seed42 experiment; no post-hoc rescue or GNN is added.

See [FINAL_REPORT.md](FINAL_REPORT.md) for all 11 metrics and 23 required answers,
`results/` for matched aggregate comparisons, and `audit/` for provenance/tests.

See `SOURCE_AUDIT.md` and `PROTOCOL_LOCK.json` first. All graph vectors, matrices,
patient/channel ledgers, clinical labels, checkpoints and execution logs stay
private on the original Windows server. Only source and aggregate audits/results
are released. This is exploratory repeatedly used development data, not sealed
prospective confirmation.

Pipeline: `inspect_source.py` and `audit_source.py` → freeze source/protocol →
`benchmark.py` → `extract.py` → `prepare.py` → `tests/engineering.py` →
`run_training.py` → `finalize.py`. All executable stages use the same protocol
and reject incompatible cached artifacts. `RAW_REBUILT` is the locked source.

Runtime: Python3.12.13, PyTorch2.11.0+cu128, RTX5090; NumPy2.4.3, SciPy1.17.1,
pandas2.2.3, sklearn1.8.0 and NetworkX3.6.1, recorded in
`audit/COMPLETION_VERIFICATION.json`. Extraction
uses four bounded single-BLAS-thread workers; formal training remains serial.
The 15 runs complete 196 epochs with 29.80 seconds total epoch-body time;
this excludes source verification, extraction, checkpoint/report I/O and tests.
No formal run crashed or was outcome-selected for repetition.

Before releasing artifacts, run `python tests/public_delivery.py` from this
directory. Private input paths are intentionally server-specific; reproducing
the experiment requires authorized access to the exact hash-locked caches and
frozen D2 banks, not downloading public patient data from this repository.
