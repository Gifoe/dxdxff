# PR-UAS seed42

Completed: 20 OOF teachers and 15 matched students across five frozen folds.
Development continuation gate **failed**; no new outer TEST was run.
Terminal: `A2_POSITIVE_BUT_BELOW_TARGET` (numerical only, no meaningful gain).

| Development metric | A0 | A1 | A2 |
|---|---:|---:|---:|
| Patient-equal Macro-F1 | .638080 | .637288 | .638229 |
| EZ-F1 | .431198 | .426363 | .428154 |
| EZ-AUPRC | .518235 | .516857 | .519669 |
| EZ-AUROC | .710667 | .707643 | .710264 |

A2−A0 Macro-F1 +.000149, 95% patient-ID cluster bootstrap CI
[-.004712, .004356]; 65 validation appearances / 47 distinct IDs.
See [FINAL_REPORT.md](FINAL_REPORT.md) for all metrics, contrasts, gate failures,
center/uncertainty diagnostics and execution limitations.

Target-only uncertainty-aware supervision for the exact historical 88D
`patient_z_mlp`. This is not an E1/E3 raw model, P23 loss, N5 PU objective or BUNDL
loss. See `SOURCE_AUDIT.md` and the immutable `PROTOCOL_LOCK.json`.

## Execution order

1. `code/reproduce_b0.py`: hash-check cache/ledger/checkpoints/folds and replay all
   original B0 outer predictions. Require exact historical metric reproduction.
2. `code/test_and_smoke.py`: synthetic tests, actual FIT-only source preprocessing
   parity and original-source one-epoch training parity.
3. `code/run_development.py`: 20 FIT-only OOF teachers and 15 matched students.
   Each teacher has a disjoint inner query and internal checkpoint-selection set.
4. `code/finalize.py`: compact patient-equal metrics, 10,000 patient-ID cluster
   paired bootstrap, diagnostics and prespecified continuation gate.
5. Only a passing development gate can admit one newly frozen exploratory outer
   evaluation. A failing gate means stop; no lambda tuning or extra losses.

All data, teacher uncertainty, channel predictions, scalers, model/optimizer/RNG
checkpoints and logs remain in the private server runtime. Public `results/` and
`audit/` contain only compact aggregates. Historical B0 outer replay is a separate
mandatory reproduction audit, never a development control.

## Running

Provide a restored original source root, original feature cache/ledger/split/
checkpoints and the historical feature manifest to `reproduce_b0.py --help`.
The replay writes `FEATURES_PRIVATE.npz` and `A0_REPRODUCTION.json` under its
`--output` (use the private runtime's `gate` directory). Then use the other entry
points' `--help`. Public cmd launchers use canonical source filenames; actual
server attempts retained versioned copies (successful trainer
`run_development_v4.py`, byte-identical to public `run_development.py`).
The original large-cache reconstruction passed in the first runtime. Replacement
runtime admission reused that exact task-generated export by SHA and freshly
replayed all historical checkpoints; it did not use another model's export.
`run_gate.cmd` documents fresh reconstruction, not an instruction to rerun a
completed experiment. Native large-cache reaggregation failures are disclosed.

The selected runtime and any compatibility work must be audited. The old wrapper
did not archive original initial weights; new A0/A1/A2 share explicitly seeded
fold-specific initial states. Frozen historical checkpoint inference, not
historical training bitwise replay, is verified.

Predictive entropy is a function of teacher confidence. Observed-label mismatch
is not independently adjudicated clinical annotation error. The matched-magnitude
constant-smoothing comparison is **target-space only**, not a fourth tuned/trained
model or evidence that correction magnitude is causally controlled for performance.
