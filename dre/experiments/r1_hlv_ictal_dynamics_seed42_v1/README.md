# R1 HLV ictal-dynamics, seed 42

This is a matched window-level R0/R1 comparison on the existing 80-patient, 7,635-channel Task-1 fixed five-fold fit/validation/test membership. R0 has the HLV branch off; R1 changes only the gated six-dimensional high-gamma, line-length and variance dynamics residual. Both retain patient-relative classification. The protocol was frozen before training in `PROTOCOL_LOCK.json`.

Run validation first with `code/run_matched.py --variant R0 --stage validation` and then the identical R1 command. Private checkpoints and patient-level predictions belong outside GitHub. The finalizer must apply the predeclared six-part validation gate before any outer-test evaluation. Prior outer outcomes on this cohort have been viewed, so even a passing result is exploratory rather than a fresh sealed confirmation.
