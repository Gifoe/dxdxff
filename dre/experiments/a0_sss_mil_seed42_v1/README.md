# A0-SSS-MIL seed42

SSS-inspired sparse raw windows + channel-level MIL, compared with exact frozen
D0/D2 engineered-feature controls. Three authorized new arms: raw-only S0,
joint hidden fusion S1 and paired channel-shuffled fusion S2.

Read `PROTOCOL_LOCK.json`, `SOURCE_AUDIT.md`, `IMPLEMENTATION_AUDIT.md` and the
aggregate `audit/` files. Formal execution is on the original Windows server.
All 15 registered runs are complete. `FINAL_REPORT.md` and `results/` contain
the verified aggregate results: D2 Macro-F1 0.647926, S0 0.611674,
S1 0.650392 and shuffled S2 0.650461. S1-D2 paired 95% CI crosses zero;
raw-disabled S1 remains 0.650355. The continuation and 0.700 target gates fail.
Terminal: `CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN`. No outer TEST evaluation
was run; this seed42 experiment is stopped without rescue variants.

Pipeline:

1. `code/audit_raw.py`: real raw/feature metadata and waveform validity gate.
2. `code/prepare.py --protocol PROTOCOL_LOCK.json --manifest audit/FROZEN_INPUT_MANIFEST.json`:
   controls replay and private label-blind raw tensor preparation.
3. `tests/engineering.py --protocol PROTOCOL_LOCK.json --device cuda`:42 checks
   plus real FIT-only smoke. Bind final code and protocol before training.
4. `code/train.py --protocol PROTOCOL_LOCK.json --device cuda`:exactly S0/S1/S2,
   five folds each, resumable one-patient updates and selected private predictions.
5. `code/finalize.py --protocol PROTOCOL_LOCK.json`:paired unique-ID bootstrap,
   all11 metrics/five folds/four centers, interventions, audits and final report.

Server paths are private execution configuration, not redistributable data.
No raw EEG, private identity/score ledgers, logs or checkpoint tensors are
included. Repeated development selection and unverified clinical onset are
explicit limitations. Stop after seed42; no rescue variants or automatic v2.
