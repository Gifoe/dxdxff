# Internal-only Raw-conditional physiological feature selection

This is a separate follow-up to the completed v2 experiment. It is not a
retrospective change to its feature sets, neural models, thresholds or Outer
results. It is motivated by the **legal Development** mismatch between a
physiology-only linear screen and full neural performance, and by the
scientific difference between standalone prediction and conditional Raw
complementarity. It does not claim an already discovered optimal feature set.

## Locked experiment

- Exact32-group admitted pool, original own-FIT patients and the same20
  internal role partitions. Original Development and Outer are excluded.
- Twenty scratch Raw-only encoders use only selection-training/calibration
  patients. Feature-choice and assessment labels cannot enter their trainer.
  A fullFIT-trained encoder on an inner held-out patient is prohibited.
- Frozen Raw representations contain both0–20/0–30-second summaries (192D),
  so physiological complementarity is not obtained merely by access to a longer
  post-reference signal interval. Both pools share one CNN state per cell.
- Conditional ridge screens all192 Raw coordinates plus selected physiological
  coordinates. Its Schur solution is exactly the full weighted ridge solution,
  independently tested against sklearn and direct solve, not a compressed
  approximation. This remains a low-cost screen, not a final neural model.
- k0–9 includes the option to select **no physiology**. Three untouched own-FIT
  assessment groups determine cardinality; no global Outer-driven consensus.
- Ten final-path patient-weight re-selections per fold estimate conditional
  subset sensitivity while encoder/preprocessing stay fixed. They do not
  establish full encoder-seed or clinical reproducibility.

`PROTOCOL_LOCK.json` fixes all choices before any new encoder training.
`conditional_io.py` checks the v2 source hashes and binds its pool, original
roles and protected score seal. All checkpoints/logs/new embeddings remain in
`D:\m8_rawconditional_feature_selection_seed42_v1_runtime`, separate from v2.
`prepare_encoders.py` redirects the imported unchanged trainer's progress output
to this new runtime; it never writes old v2 state.

## Execution

On the trusted server, copy source to
`D:\m8_rawconditional_feature_selection_seed42_v1`. Then keep the authenticated
foreground owner alive while running `run_internal.ps1`. It preserves uniquely
named logs, takes an exclusive lock, admits actual role isolation, performs the
20 encoder fits, then runs the CPU conditional screen. On an engineering
failure inspect logs first and use exact checkpoint resume only. Never start a
duplicate live worker or touch unrelated GPU jobs.

GPU runtime/source/budgets remain those of v2. CPU ridge/statistics use the
existing stable Python3.10/NumPy2.2.6 environment and process-local affinity16.
Only the two changed ridge algebra/preprocessing tests are added. No repeated
toy model is substituted for real full-patient Raw training.

Local algebra tests:

```powershell
python -m pytest experiments/m8_rawconditional_feature_selection_seed42_v1/test_conditional_ridge.py -q
```

## Output and stop boundary

Only actual anonymous source/role/encoder admission counts, conditional search
records, cardinality/old7 assessment metrics, feature allocations and stability
diagnostics may be published. Private patient arrays, labels, embeddings,
weights, scalers, logs and raw data remain on the server.

This registration does **not** authorize new Outer inference or automatically
train a rescued final M8. A useful conditional subset, if actually supported,
requires a separately fixed model-family confirmation protocol. A zero-feature
or unstable result is reported as such, not followed by unbounded search.
The full active research goal remains unproven until sufficient evidence exists.
