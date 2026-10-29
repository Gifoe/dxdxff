# Executed verification

2026-10-29, Asia/Shanghai. Main-table-matched A-C only; no model fitting.

- Frozen analysis source e631d60f801bfda5ca46e59f8fd85d3fd7c39f55 was committed
  and pushed before the new analysis. All parent scientific source pins passed.
- Five selected residual heads bind the same five original ictal TimeConv
  checkpoints as the existing No PR condition. Parent selections and score seals
  passed admission; all original artifacts remained hash-identical.
- Eighty patient-specific fixed-head replays used the complete frozen original
  representations and exact original population-Z implementation. Maximum
  absolute probability error was 0.0 against the main-table sealed scores.
- All 45 selected cells froze and all 45 score payloads sealed before new
  endpoint reconstruction. No thresholds, backbone parameters or head parameters
  were changed. Forty score cells were reused unchanged; five residual score
  cells replaced the end-to-end PR cells in this independent version.
- All 80 patients and 7,635 channels are included in each of nine conditions.
  There are 720 endpoint rows, 4,320 budget rows and 480 patient curve rows.
- All 11 aggregate TimeConv Original/PR Linear Residual endpoints reproduce the
  existing main table within 1e-12. The 640 other patient endpoint rows match the
  historical A-C rows within 1e-12.
- Independent tie-aware AUROC/AP formulas and deterministic channel tie order
  checks pass for every patient. Score-Z AUROC/AP/MRR/Hit@1/3/5 equal No PR exactly.
- All 1,203 overall, paired, review-budget and curve intervals were independently
  recomputed by direct resample indexing; primary count-matrix bootstrap and
  the independent implementation agree within 1e-12. The frozen comparison
  families contain 6, 12 and 9 adjusted intervals, respectively.
- Source-weighted means reproduce patient-equal means; four-source means and
  minima reconstruct CM and Worst. These are source-grouped outer results, not
  LOSO results. Budget curves are monotone from 0 to 1 and agree with the
  reported 5%, 10% and 20% review table.
- All 480 patient-wise paired-difference rows are saved in trusted private
  storage only, with a separate private reporting seal. None is exported.
- Whole public archive SHA256 and every allowlisted payload hash passed before
  extraction. Only anonymous aggregate tables and metadata were downloaded.
- Five CPU semantic tests passed. Local source/rank/budget/paired checks passed.
  PNG previews were visually inspected; PDF/SVG/600 dpi PNG exports are provided.
- All staged anonymous payload blobs retain their original bytes and hashes.
  A real Git-index tree archive passed the same five CPU tests, aggregate
  consistency checks, frozen-function AST parity and figure-format audit.

Public archive SHA256:
`f26fab5a7b04cc5e680abb52ef4ac43dc4eff5618043ea6748b32f62e4d62aa5`.

The original end-to-end A-D results, including seizure sensitivity D, and all
existing JBHI main-table values remain unchanged. Exploratory interpretation
retains nonpositive source changes and intervals spanning zero.
