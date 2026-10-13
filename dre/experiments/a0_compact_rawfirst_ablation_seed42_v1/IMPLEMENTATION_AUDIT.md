# Implementation contract

- `dependencies.py` imports/hash-checks original source-free constructor,
  trainer/core, frozen feature archive/order/splits and legal FIT/VAL banks.
- `audit_sources.py` independently admits authoritative F0 and restoration
  controls F6/F7; verifies exact models/masks/preprocessors/selection; reuses
  historical sealed scores to check old80-patient aggregates; then binds the
  entire new code/test/protocol matrix before new raw outcomes.
- `raw_alignment.py` uses hash-pinned RawAlignmentStore plus stricter valid
  sample interval checks. All waveform banks and identity ledgers stay private.
- `raw_only_model.py` instantiates exact RawTinyEncoder and the original
  **initial** A0 shared network. Missing windows/seizures are excluded in
  two-level pooling; population std is zero for single-seizure channels.
  Per-patient latent normalization uses all canonical channels and SD floor1e-8.
  Row indices only retrieve waveforms/metadata and do not enter learned layers.
- `training.py` delegates unchanged loss/update/optimizer/RNG/selection to
  the original trainer. R0 exposes indexed waveform batches; feature arms
  expose their exact masked inputs. No raw-head/source weights are restored.
- `evaluation.py` saves validation replay and locks every stage selection,
  preprocessor, checkpoint and code before any new outer inference. Whole-stage
  scores are sealed before the global clinical-y member is read. Native
  interruption can reuse only existing matching atomic score files.
- `verify_completed.py` never runs a model. Independent sklearn/ranking and
  bootstrap-multiplicity calculations verify sealed-score metrics, aggregate
  estimands, fold/center tables and fixed-threshold contrasts.
- Stage III requires verified Stage II results and a recorded verified public
  raw-result commit. Raw performance cannot alter this gate or the fixed matrix.

Tests were executed before clinical raw training. Raw input purity, pooling,
mask/gradient and initial-state tests PASS; exact CPU model/optimizer resume
and CPU/CUDA RNG restore PASS. GPU arithmetic bitwise resume is not claimed.
Two preclinical fixture fixes and unchanged tolerances are disclosed by
`PRECLINICAL_ENGINEERING_AMENDMENT.json`, the numerical diagnosis and raw report.
All30 new competitive cells complete,333 epochs; both independent stages and
the reporting-only nine-arm union PASS. `verification/finalize_aggregates.py`
was added only for reporting after outcomes; scientific frozen code is unchanged.
No installed skill was used.
