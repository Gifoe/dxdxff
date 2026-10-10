# Execution and preservation ledger

Trusted Windows server: the original user-authorized SSH password-authenticated
entry point `remote_exec_safe.py` was used. No askpass script or new credential
storage was created. Python 3.12.13 / PyTorch 2.11.0+cu128, CPU-only stored-score
analysis; no GPU training or scoring job was launched.

The historical source review, 101-file snapshot, new protocol and code were
written before the first density outcome. Twelve synthetic/parity/FIT smoke
tests passed before analysis. Core code/test hashes are in ENGINEERING_TESTS.

Two disconnected `Start-Process` launches did not remain active after the
SSH session ended. The first produced no phase output; the second stopped after
partial FIT progress. No Python exception or relevant Windows Application Error
was observed. Session-lifetime termination is the observed operational issue,
not proof of a density/runtime numerical failure. Logs remain on the server.
No outcome tuning, code change, scorer replay or checkpoint alteration followed
these incomplete launches.

The connected execution then completed the unchanged audit in 28.436 seconds,
return code zero. All twenty density checks and all four sets of 255 LOO
appearances completed. All four admissions failed on decision utility, not
numerical validity or severe collapse. No posterior VAL inference was run.

A separate unchanged-protocol/seed execution in a new private repeat directory
completed independently. Verification compared 22 public files byte-for-byte,
both sealed private phase objects recursively, engineering-test bytes, and all
101 original private hashes. Everything passed. The two incomplete launches
were not treated as scientific replicates, successful folds or selectable runs.

Only code, protocol, hashes and compact non-identifying aggregates were fetched.
No raw data, model tensors, private predictions/mixture fractions, patient or
channel records, clinical labels, OOF ledgers or runtime logs were downloaded
or published. There was no outer TEST read/evaluation and no follow-up model.
