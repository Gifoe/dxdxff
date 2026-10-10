# Frozen A0/D2 robust posterior identifiability — seed42

Completed terminal: `NUMERICAL_REPAIR_WITHOUT_DECISION_GAIN`.

All 20 full-FIT density checks passed. All four arms completed 255 FIT-LOO
patient appearances, but MAP reduced Macro-F1 relative to the identical
fixed-prior decoder in every outer-FIT group:

| Arm | FIT MAP Macro-F1 | FIT fixed-prior Macro-F1 | Difference | MAP prevalence MAE | Fixed-prior prevalence MAE |
|---|---:|---:|---:|---:|---:|
| D1_R0 | 0.611159 | 0.625864 | -0.014705 | 0.142021 | 0.117946 |
| D1_R1 | 0.618811 | 0.625752 | -0.006941 | 0.132774 | 0.117946 |
| D3_R0 | 0.601599 | 0.623724 | -0.022125 | 0.160460 | 0.117946 |
| D3_R1 | 0.618009 | 0.626703 | -0.008693 | 0.130576 | 0.117946 |

No posterior arm was admitted to VAL; its full development performance is
`NOT_ESTIMABLE`, not zero. Frozen development controls reproduce A0 Macro-F1
0.638080 and D2 0.647926 exactly. Their old +0.009846 difference retains the
95% patient-cluster interval [-0.004723, 0.026640]; it is not a new posterior gain.

R1 full-FIT clipping was 0% in every fold and source. Numerical repair succeeds,
but does not reveal useful MAP decision information under this frozen Gaussian
family. This supports the numerical explanation for the old invalid moments,
while failing to establish patient-mixture utility. It does not establish that
all physiological signals or all possible unlabeled distribution models lack
information. FIT-LOO is a screening diagnostic, not fully nested validation,
because other OOF teachers can have seen the pseudo-target's labels.

All 22 public runtime files are byte-identical on an independent unchanged-seed
repeat, and both private phase objects match exactly. All 101 predecessor
private hashes remain unchanged. See [FINAL_REPORT.md](FINAL_REPORT.md),
[audit/ENGINEERING_TESTS.json](audit/ENGINEERING_TESTS.json) and
[audit/DETERMINISM_AND_IMMUTABILITY.json](audit/DETERMINISM_AND_IMMUTABILITY.json).
No rescue tuning or automatic follow-up was performed.

## Reproduction and artifact layout

New independent diagnostic protocol; the predecessor is not modified. Four
posterior candidates D1_R0/D1_R1/D3_R0/D3_R1, two immutable threshold controls
D0/D2, no scorer/teacher training and no outer TEST evaluation.

Run `code/run.py --runtime <new private runtime> --previous <old runtime>
--source <server experiment base> --protocol PROTOCOL_LOCK.json` using the
original compatible Python312 environment. Run tests/test_contract.py first
with the same protocol and source arguments. Gate hashes bind inputs and code.
Each phase seals private artifacts before the next; no failed-fold partial
mean is used as a complete arm. A separate exact repeat verifies public-output
determinism. Only code and non-identifying aggregate artifacts are committed.

Public CSVs are in `results/`, public gate/integrity JSONs in `audit/`, and the
sealed `RUN_STATUS.json` and `FINAL_REPORT.md` at the experiment root. The status
hash index uses the basenames of the trusted server's flat `public/` directory;
these bytes are unchanged by the repository layout. Private `.pt` objects,
checkpoints, individual records and execution logs were not transferred.
