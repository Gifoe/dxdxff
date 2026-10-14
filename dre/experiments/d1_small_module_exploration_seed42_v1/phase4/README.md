# Patient-observable decision adapters — prospective Phase4

User requested whether the completed Development0.666795 can exceed0.70, followed by Outer. This phase does not reinterpret the completed M8 Outer0.634097 as0.66, and never uses its patient-level outcomes to fit/select a new model. Previous results and freezes stay unchanged.

Before new training, three candidates/fifteen original fold cells are locked. The new evidence bank runs the selected frozen M8 only on each original FIT/validation cohort, retaining exact validation score parity. The bank includes processed14D physiology, frozen96D morphology and label-free patient score distributions; no additional Raw/EDF/physiology extraction. Frozen evidence can be reused, so repeated adapter epochs do not re-encode all waveform windows. FIT-only patient-equal feature normalization and original validation threshold/checkpoint selection are retained. No identity or center feature enters the adapter.

- M9 learns one patient logit shift from16 observable context statistics; ranking within a patient is unchanged.
- M10 learns a bounded zero-initialized correction from129D channel+patient evidence.
- M11 has the same architecture as M10 with the prospectively specified stronger/temperature-sharpened SoftMacro objective.

All models equal frozen M8 at initialization. Each patient receives one supervised update per epoch, and original own-fold TEST identities are absent from the bank. A label-derived threshold ceiling near0.729 is a diagnostic upper bound, not a model metric or legal threshold rule. A fixed top20% channel heuristic scores only0.637, so prevalence/count forcing is not justified.

Server entrypoints: `code/test_models.py`, `code/bank.py`, `code/train.py`, then `tools/verify_development.py`. Read `PROTOCOL_LOCK.json` for fixed loss/caps/selection. Existing D1_PHASE1_CODE/D1_PHASE2_CODE/DGM_COMPLETED_CODE/DGM_BASE_CODE paths and OMP/MKL/OpenBLAS thread limits1 are required. Sources/models/banks/thresholds and complete epoch state are hash-bound; changed scientific code is rejected under an existing freeze.

No achievement is asserted until all15 cells and independent verification complete. The goal remains active if0.70 is not reached; do not silently lower it. Only after a final model is selected/frozen will the separately authorized Outer run. All additional Development attempts are adaptive, repeatedly viewed and selection-conditioned, not independently replicated. Historical Outer exposure and overlapping cross-fold Development IDs rule out a claim of blind clinical confirmation.

Only code and anonymous aggregates may be published; private banks, checkpoint/scalers, labels/identities, individual scores, Raw and logs stay on serverD. A first transfer attempt lacked the new code directory, before runtime freeze/training; the directory was created and sources retransferred. One synthetic test had a tensor/scalar helper TypeError, corrected before server freeze; no scientific algorithm was changed.
