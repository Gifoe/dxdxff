# Raw-Delta Encoder — Seed42

RequestedD1–D3 with required matchedE1 control and historicalD0 Development replay. Original80/256/7635 cohort and fixed5 folds; **no new Outer authorized**. No installed skill used.

StageA first rechecks exact caches, named/compound alignment and every500-sample genuine Raw slice, then reconstructs historical all-pre DELTA and tests E14 archive parity and fixed4/5 A/B-family Spearman gate. Original PSD/RMS/entropy formulas and bandwidth restrictions are preserved. A failed gate is a scientific terminal, not permission to alter bands, features or train a larger network.

Only passing StageA permits matched1537-parameter E1/D1 training and frozen originalDELTA checkpoint identity. Only passing predictive recovery permits frozenD1+smallCNN/GRU D2/D3 training and interventions. Future-stage files are implemented only when their admission gates pass; no unexecuted metrics will be populated.

Trusted-server first-stage command: `code/prepare.py --runtime NEW_PRIVATE_RUNTIME --source PINNED_HISTORICAL_SOURCE_ROOT`. Protocol and reconstruction source freeze before measurements. FIT-only numerical fidelity uses actual79 unique originalFIT IDs; all80 remain in data coverage. Development65 appearances/47 identities is separate from this label-blind FIT gate and from historicalOuter.

Completed: Stage A and B PASS; all D1/D2/D3 five-fold Development runs and matched E1 completed. Final terminal: **RAW_INFORMATION_POSSIBLY_USEFUL_TEMPORAL_ORDER_NOT_ESTABLISHED**. Primary AUROC/Macro-F1: D1 0.725978/0.644546; D2 0.732515/0.650238; D3 0.727976/0.649008. D2 misses both registered AUROC minimum-effect gates. See [FINAL_REPORT.md](FINAL_REPORT.md) for all metrics, cluster intervals, interventions and limitations.

Publish only source, hashes, anonymous aggregates and reports. Raw, identity/clinical fields, private reconstruction tables, weights, predictions and logs remain on the trusted server. No new Outer run was performed.

## Reproduction sequence

On the trusted server, run against the hash-pinned historical source root, a new private runtime, and the original caches. `--source` defaults to that server's existing historical code root.

1. Run `python -m pytest tests -q`.
2. Run `python code/prepare.py --runtime PRIVATE_RUNTIME`. Stop unless `results/STAGE_A_GATE.json` passes.
3. Run `python code/train.py --runtime PRIVATE_RUNTIME`. This replays D0, checks compact/preprocessing parity and trains E1/D1. Stop unless `results/STAGE_B_GATE.json` passes.
4. Run `python code/raw_store.py --runtime PRIVATE_RUNTIME` to reverify existing waveform-only banks against the original Raw source and exact metadata.
5. Run `python code/train_dynamics.py --runtime PRIVATE_RUNTIME` for all D2/D3 folds, interventions and cluster intervals.
6. Run `python code/verify.py --runtime PRIVATE_RUNTIME` for independent aggregate/checkpoint verification without new model inference. `metadata_diagnostics.py` provides the label-blind sample-unit diagnostic separately.

Original protocol and source hashes are frozen before their corresponding stages. On native interruption, only identical-binding atomic completed patients/epochs may resume; never overwrite the freeze to accept changed scientific code. The observed NumPy access violation was avoided on the successful run with OMP/MKL/OpenBLAS thread limits1. Python3.13.9, NumPy2.3.5, pandas2.3.3, sklearn1.7.2, PyTorch2.11.0+cu128 and RTX5090 were used. The locked protocol is the authority if notes or defaults differ.
