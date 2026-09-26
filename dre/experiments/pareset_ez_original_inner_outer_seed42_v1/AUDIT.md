# PaReSet-EZ v1 server pre-training audit

Status: **preflight passed; no performance conclusion yet**. This is an isolated implementation in `E:\DRE-nips\new-pipeline\7-11\pareset_ez_v1`, not a modification of prior EpiLENS outputs.

## Frozen development data and protocol

- Formal cohort: 80 patients (HUP 36, LZU 21, multicenter 15, pediatric 8), 256 seizure runs. All 80 occur in the existing feature cache and frozen five-fold ledger.
- Frozen fit/validation/test membership is `D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv` (SHA256 `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278`). Fold 1 contains 51 fit, 13 validation and 16 test patients. Test membership was checked; test outcome was not evaluated.
- Source feature cache: `D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl` (SHA256 `9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087`). It has 28 features, not the 36-D supplementary input. We select nine features by verified names and generate four views using the supplied implementation. This yields 36-D model input; it does **not** make the new experiment identical to the historical 28-D CDEL experiment.
- Window centers are explicit seizure-onset-relative seconds; 253 runs have 59 centers from -29 to 29 seconds and three runs have shorter explicit arrays. Every selected run has pre- and post-onset windows. The cache records 2-second feature windows and 512-Hz source sampling. Feature names, channel alignment, finite arrays and monotonically increasing times passed inspection.
- Formal labels in `patient_index.labels` are EZ-positive (1=EZ). They are converted **once** to model NEZ-positive (1=NEZ). Run-level copies disagree with that canonical source for 85 channels in two LZU patients; the original server loader uses the patient index. The adapter follows that loader and records this discrepancy. This is a data-quality caveat, not a label-based adjustment.
- The train-only standardizer is fitted on the 51 fold-1 fit patients. Validation and test data are transformed with the same fit statistics; neither is used to fit the standardizer.
- The raw cache available on disk has been resampled to 250 Hz, so it cannot regenerate 80–150-Hz high-gamma features at the original 512-Hz resolution. The existing verified feature cache is therefore the source for this experiment.

## Model and numerical tests

- Supplied `PaReSetEZ(ModelConfig())` has 18,474 declared parameters. The user package's 33 tests pass on the server. The copied retained base has maximum no-Q10 output difference `3.58e-7` in the supplied audit.
- Module counts are encoder 13,096; reference 576; phase 3,548; adaptive seizure pool 1,221; final EZ head 33. Thus the base variant has 13,129 active parameters, no-pool 17,253, no-reference and uniform-reference 17,898, while full and BCE-only have 18,474. The unused modules remain allocated in some ablations but are not counted as effective capacity.
- Real-data fold-1 preflight: a 102-valid-channel patient, tensor shape `[3, 59, 102, 9]`, four-view expansion to 36 dimensions, GPU forward/backward 3.42 seconds including first invocation, peak CUDA allocation 149.6 MB, finite loss and gradients. This was a single diagnostic pass, not training or performance evaluation.
- The supplied supplementary PRQ/BCR code has a nonfinite single-seizure backward path through `sqrt(variance)` (recorded in `SOURCE_AUDIT.json`). The actual server `neuroez_c/p23_seizure_tail.py` already applies `variance.clamp_min(1e-8).sqrt()`. A separate real-server ATC7 tail regression passed finite output, input gradient and parameter-gradient checks for one valid seizure and for three identical seizures (`SERVER_NUMERIC_TEST.json`). This does not prove every historical PRQ/BCR branch is numerically safe.
- The server's historical PRQ/BCR/CDEL system is built from its own 28-D feature pipeline and training plans. The 0.8/0.2 CDEL is the paper-relevant combination. A separate 0.9/0.1 exploratory fusion is not substituted for it. Historical metrics are contextual, not a controlled PaReSet architecture delta.
- The observed historical plan is not one universal schedule: paper-producing PRQ used 45 maximum epochs, minimum 18, patience 6, while BCR used 40 maximum, minimum 18, patience 8. The *new matched development reruns* deliberately share the prospective 45/18/6, AdamW LR `1e-4`, weight decay `1e-3`, dropout `0.4`, four-patient batch schedule. Thus they are matched new controls but not literal reruns of each component's distinct historical schedule. The supplied supplement defaults (30 epochs, minimum 6, weight decay `1e-4`) were not used as a proxy for the server plan.
- Matched 36-D supplementary PRQ/BCR controls use a local `masked_mean_std` repair with an exact-zero forward at zero variance and finite backward. The repaired implementation matches the unmodified supplementary function exactly on a nondegenerate random regression case (max absolute difference 0.0) and passes single- and three-seizure finite-output/gradient tests for both branches (`CONTROL_NUMERIC_TEST.json`). This is newly run control code, not the historical paper checkpoint.

## Source and data hashes

| Artifact | SHA256 |
| --- | --- |
| `pareset_ez.py` | `fe452ba76f0c3ae65c963e00a7dd51a6d6750752b3a024d699b34af53900e72f` |
| `train_with_supplement.py` | `e14bf84367c7e2d2d46976e0bc4bfb0a9111133b0d274167f7eaa6d71c145b5b` |
| `build_patient_records.py` | `bd326331accd3b7af74878988319956ac7c6f23b460968ea4a1cc8936c46b93d` |
| supplied `epilens/models.py` | `d6cbb4bc4a951623c59462b3ee09f49b16b1fea6dd9cf02ec49be121c06136ef` |
| frozen partition manifest | `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278` |
| adapted 9-D patient records | `e4cb1347cdc343d6aef288cdcb701c0e7575d71dcdde485e55157465134a77e6` |

No test score, surgical outcome, true-K or patient/center ID is passed to inference. No privacy-bearing cache or checkpoint belongs in a public code/results commit.
