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

## Executed outcome

The actual conditional study is complete and independently checked at the
patient-metric/role/selection level. See [REPORT.md](REPORT.md),
[ENGINEERING_AUDIT.md](ENGINEERING_AUDIT.md) and
[the project-wide synthesis](../m8_physiology_goal_evidence_audit_seed42_v1/RESEARCH_SYNTHESIS.md).
Twenty Raw encoders completed286 epochs. The accepted full-reference screen
contains9344 head fits, plus25 distinct final-choice control fits. The initial
9344-head block screen remains unaccepted because its actual score gate failed.

Chosen cardinalities are1,0,9,2,1; final fold-set mean Jaccard is0.0. No new
global feature set or neural clinical improvement is declared. The positive
conditional internal contrasts are selection-conditioned and never replace
the original neural Outer comparison. No new Outer was run in this follow-up.

### Reproduce the already completed trusted-server artifacts

The original scientific files, input freezes and numerical amendment are kept
byte-identical. `run_internal.ps1` and `run_dense_reference.ps1` document the
historical execution, including the failed shortcut and terminal metadata
exception. Do not launch either against an already completed experiment.
With the original licensed private artifacts installed at their pinned runtime
locations, use the terminal recovery/check entry:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u `
  'D:\m8_rawconditional_feature_selection_seed42_v1\resume_reference_checked.py'
```

It refuses a partial screen, checks original bindings, fills missing canonical
admission/terminal metadata and verification only when needed, then completes
the registered choice controls without neural training. Completed scientific
screens are not replayed as new evidence. For an explicit full9179+165-head
verification replay, run `verify_and_report_internal.py` under the same CPU
environment after inspecting worker state. Keep the SSH foreground owner
alive; never create a duplicate running worker.

The public-only workflow requires NumPy, pandas and matplotlib, and does not
require identifiers, private representations or GPU data:

```powershell
python experiments/m8_rawconditional_feature_selection_seed42_v1/figures_internal.py
python experiments/m8_rawconditional_feature_selection_seed42_v1/verify_public_release.py
```

`results/dense_reference/` is the accepted canonical output. Earlier failed
numerical admissions remain directly under `results/`; do not merge their
status or pass them off as accepted results. Figure CSVs preserve all k0–9,
all32 feature-frequency rows and all source strata. The original study's
135-file release is also checked by the public release verifier.
