# Zero-shot patient geometry recovery (development study)

The SHA-256 locked protocol is in `PROTOCOL_LOCK.json`. The source is exact A1
(`b2871b32873b67d0e6155d2766ef040ee1e9df01`). Training starts from the
same A1 architecture and per-fold initialization; Z1/Z2/Z2B/Z3/Z4/C1 learn
the representation and original classifier jointly. No target-patient update,
candidate pool, or B=8 support is used by the new models.

`DIAGNOSTIC_AMENDMENT.json` was locked before new target outcomes. It sets a
common epoch-30 checkpoint per fold/variant for R4 geometry diagnostics;
mixing per-patient VLOO-selected checkpoints would compare nonshared spaces.
The primary VLOO performance protocol is unchanged.

On the original Windows server, private checkpoints, optimizer states, score
grids, R4 tensors and patient/channel records live only under `ZSG_RUNTIME`.
The compact aggregate CSV/JSON/MD reports live in this directory. The runner
requires the existing A1, SRGI and active-fewshot private caches, plus the
unchanged source modules and cache hashes checked by `train.py`.

Run order: `test_geometry.py`, `train.py --stage audit`, `drive.py --stage all`,
`evaluate.py --stage freeze`, then `evaluate.py --stage evaluate --fold F
--variant V` for every fold and new variant, then `finalize.py`, then
`validate.py`. FIT-only selection must finish before full-FIT training, and
**all** new-model score grids must be hash-frozen before label-using VLOO.

The A1 VLOO rule has cross-patient label dependencies: a patient's label is
excluded from its own model selection but contributes to other patients'
selection. Thus literal strict target-label sequencing is not attainable with
the exact inherited VLOO rule. The legacy loader also materializes all 80
labels, although this study computes no outer prediction/metric/selection.
These limitations are explicit in the final audit, not treated as sealed test.
