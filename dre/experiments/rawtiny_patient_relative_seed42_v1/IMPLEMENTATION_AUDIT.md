# RawTiny Stage A implementation audit

- Exact A1 source and split hashes were checked before any new target-outcome evaluation.
- The repository's `RawAlignmentStore` matched all 80 A1 patients, 256 records, 24,995 run-channel incidences and 1,471,965 windows. There were zero non-finite raw values.
- Input is the existing 250 Hz, 60 s raw cache, cropped at the feature cache's 2 s window centers into 500-sample windows. No EDF recut or extra filtering was introduced.
- M1/M2 use the same RawTiny window encoder and deterministic per-fold initialization. M1 has no channel context. M2 uses absolute and patient-relative 64D representations, an exact forward percentile with smooth-rank straight-through gradient, and one permutation-equivariant attention block.
- M3 loads the fold's A1 epoch-30 FIT-trained checkpoint, freezes that pathway, then adds a zero-coefficient raw window residual and zero-output patient-relative logit residual. On a real fold-1 FIT batch, alpha=0 yielded maximum logit error 0 and zero prediction mismatches.
- Synthetic permutation and shape tests passed; real FIT-batch gradient checks found finite gradients in all three models.
- Trainable parameter counts: M1 5,729; M2 32,993; M3 34,018. These are well below the prompt's approximate 100K–300K target. This is a smaller capacity test and must not be misrepresented as matching that suggested range.
- The raw cache lacks an explicit onset-centered flag. Construction code centers the seizure onset in the 60 s raw waveform, and every matched record has onset/valid-interval fields, but independent EDF-to-cache temporal provenance was not verified. The report must retain this limitation.
- The legacy A1 loader materializes validation labels while constructing datasets. New score snapshots must not index labels before their frozen score manifest; literal strict no-target-label materialization is false.
- Exact A1 VLOO checkpoint selection uses other validation patients' labels. It never uses a patient's own label for its own checkpoint, but it is not strictly FIT-only epoch selection. This conflict is explicit in `PROTOCOL_LOCK.json`.
