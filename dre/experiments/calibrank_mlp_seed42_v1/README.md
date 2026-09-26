# CalibRank-MLP, exploratory seed 42

This experiment tests whether a tiny patient-level calibration head and a bounded channel-ranking residual can improve the historical patient-relative MLP (B0). The frozen 80-patient Task-1 fit/validation/test membership and the original 88-feature input are reused. Outer outcomes had been viewed in earlier work, so **this is exploratory, not fresh confirmatory validation**.

`code/baseline_diagnostics.py` audits historical B0 predictions and retrospective oracles. `code/prepare_b0_embeddings.py` replays the five original B0 checkpoints and extracts frozen hidden states/logits into private server-side caches. `code/run_calibrank.py` trains B1/B2/B3 using fit patients, selects each fold's checkpoint and threshold on validation patients, and evaluates the corresponding outer test once. Epoch 0 exactly reproduces B0 and remains an eligible fallback.

The public `results/` files contain only protocol information and aggregates. The server retains private per-patient/per-channel tables, embeddings, cache, checkpoints, and logs; none belong in Git.

See `BASELINE_AUDIT.md`, `ORACLE_DIAGNOSTICS.md`, `MODEL_DESIGN.md`, and `REPORT.md` for provenance and interpretation. `FINALIZATION_REPAIR.md` documents the post-training aggregate-only repair; `code/finalize_calibrank.py` is the reproducible finalizer. `code/uncertainty_diagnostic.py` produces post-hoc, aggregate-only uncertainty tertiles.
