# Compact A0 → Raw-only → matched ablations, seed42

User-defined complete F0 is the authoritative source-free, no-ZDELTA model
from `d2_no_source_no_zdelta_outer_seed42_v1`, not source-free all88D A0.
F0 has a masked88D interface with62 retained ABS/DELTA/RATIO columns, native
LayerNorm and8,817 parameters. The entire requested matrix is fixed before
new raw outcomes. F6/F7 are restoration controls, not deletions.

Execution order: historical identity/reuse audit and matrix freeze → one
new five-fold waveform-only R0 → frozen outer and independent verification
→ publish the completed raw report →25 new compact F1–F5 cells → frozen
outer/verification → final separate raw-baseline and component tables.

Raw-first stage is complete and independently verified. R0 outer AUROC
0.633626, AUPRC0.408496, Macro-F10.543958 versus F0's
0.736977/0.530653/0.624236. See [RAW_ONLY_REPORT.md](RAW_ONLY_REPORT.md).
Feature F1–F5 training remains NOT_RUN at this first raw-result publication.
Executed status and results are in `results/`; unfinished arms remain NOT_RUN.
No raw outcome can select the next arm. No installed skill is used.

## Trusted server execution

The scripts require the exact private historical banks, sources and seals.
Public data are not distributed. Use the pinned original Windows runtime;
the new runtime defaults to `D:\a0_compact_rawfirst_ablation_seed42_v1_runtime`
because C has insufficient spare space for a new waveform bank.

1. `code/audit_sources.py`: admit F0/F6/F7 and freeze the entire protocol.
2. `code/verify_feature_masks.py`: build/audit legal FIT/VAL inputs, no training.
3. `code/raw_alignment.py`: whole-cache hashes, exact compound/channel/sample
   matching and fully valid2s windows; private waveform-only patient banks.
4. `tests/contract.py`: executed numerical, purity, masking, initial-state,
   chunk/gradient, exact CPU optimizer/model resume with CPU/CUDA RNG restore,
   and single-FIT label-swap checks. Bitwise GPU arithmetic resume is not claimed.
5. `code/raw_only_train.py`: exactly5 new R0 model/fold cells.
6. `code/raw_only_outer.py`, then `code/verify_completed.py raw`: freeze all
   selections, score-only outer, seal all new scores, then full-y evaluation.
7. Publish verified raw results and record its verified GitHub SHA with
   `code/record_raw_publication.py COMMIT_SHA`. Feature training is blocked
   until this stage is verified/published, regardless of raw performance.
8. `code/ablation_train.py`, `code/ablation_outer.py`, then
   `code/verify_completed.py feature`. No matrices or thresholds change.

New training uses the unchanged original A0 trainer, including R0 through
an indexed waveform adapter. Learned layers receive no index, identifier,
engineered feature, label or center. Indexing only retrieves complete patient
waveforms. Labels are separately supplied only to legal FIT/VAL loss/selection.

The historical-cache pickle format materializes other fields at deserialization;
the raw reader discards engineered values/label fields without indexing them.
New raw training never indexes the global feature-NPZ y member. Stage I's
historical sealed-score provenance audit reads historical y explicitly, in
a separate process, as requested. New outer processes first seal every new
score, then read that member. This is not a claim of physically label-free
legacy serialization or fresh blinded test data.

Public scope is source, hashes, non-identifying aggregate metrics/audits only.
All raw arrays, banks, identifiers, per-channel/patient scores, checkpoints,
optimizer states and runtime logs remain on the trusted server.
