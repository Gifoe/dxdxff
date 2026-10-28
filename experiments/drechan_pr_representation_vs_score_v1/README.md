# Patient-relative representation versus output alignment

Independent, fixed supplementary experiment: 12D Linear SVM, original TimeConv-CNN,
and the final six-feature DREChanNet without guidance or head interaction.
All original five patient splits, 80 patients, 7,635 channels, 256 seizures and
1,471,965 valid raw windows are retained. This is exploratory repeated-cohort
evidence. There is no run, patient, feature or outcome-dependent stopping rule.

Seven conditions per model: NoPR, Global, Center, Scale, FullPR, Score-center,
Score-Z. Existing NoPR/FullPR are read-only references, SVM Global is an explicit
NoPR alias, and output alignment fits no new model weights. Expected work is
40 main fits, 30 new SVM OOF auxiliaries and 105 logical selected cells.

`PROTOCOL_LOCK.json` fixes all equations, parent conventions, initialization,
statistics and interpretation. The final model changes both reference branches
together. Neural TRAIN moments are computed in eval/no-grad at initialization
and after each complete epoch, preserving RNG, training modes and BN buffers.
The global moments associated with each selected checkpoint are sealed with it.

The SVM uses the original fold-specific C values and patient-OOF Platt procedure.
Every new inner SVM split re-estimates its view/global moments, imputer and scaler
on sub-FIT only. Physiological views retain the original downstream TRAIN scaler.
Global has the canonical existing NoPR TRAIN-scaler implementation, without a
redundant second standardization. Numeric support follows each pinned parent.

Output alignment uses native EZ logits, not clipped inverse probabilities.
The frozen NoPR checkpoint is retained, and only its validation threshold is
reselected. Each family retains its actual parent grid, orientation and tie
policy; DREChanNet's original NEZ threshold policy includes its pooled secondary
tie value. Ranking endpoints retain the exact NoPR anchor, with independent
affine-order validation, to avoid treating sigmoid rounding as ranking gain.

Run using the pinned server environment and exact source commit:

```
python entry.py admit
python entry.py all --source-commit <frozen-commit>
```

Execution writes only the new canonical task runtime under `D:\Codex_Research`.
`path_adapter.py` maps legacy paths inside this process and refuses writes to
frozen parent experiment directories. No D-root compatibility links are created.
Source and protocol hashes must match admission; all selections freeze before
new test scoring, all scores seal before endpoint computation, and independent
metrics/bootstrap verification precedes release. A single coordinator owns all
workers. Interrupted fits preserve complete-epoch optimizer/model/RNG state.

Public exports contain only the explicit aggregate allowlist and manifest.
Validate `release_check.py --package <archive>` before allowlist extraction.
Signals, identities, patient/channel endpoints, logits, embeddings, reference
parameters, scalers, weights, optimizer state and runtime logs remain private.
Existing JBHI tables and figures are not overwritten.

## Completed execution and disclosure timing

All 105 logical cells, 40 new main fits and 30 SVM OOF fits completed with
independent verification of 1,680 patient-condition rows and 15,120 endpoint
values. Final results are in `results/RESULTS_ZH.md` and `results/public/`.

During execution, the user explicitly requested the SVM/DREChanNet outer results
before TimeConv finished. Their 70 cells were separately frozen, scored, sealed
and independently verified before that early disclosure. The original common
105-cell disclosure timing was therefore amended; subsequent TimeConv training
and selection used the unchanged frozen source and rules. The final 105-cell
freeze/score-seal statements apply to the primary pipeline, with the earlier
70-cell preview separately disclosed in the public freeze status and findings.
Immutable private freeze/score seals were preserved. Final GPU evaluation
matched every one of the preview's 10,080 patient endpoint values exactly.
