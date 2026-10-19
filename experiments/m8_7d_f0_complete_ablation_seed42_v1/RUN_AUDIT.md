# Actual execution and release audit

## Fixed source and admission

The admitted scientific source was pushed to GitHub before formal fits at **c8cd777d5b84d6cd7f17ebad1649aafc7dcfd537**. The eleven source/protocol file hashes in [SOURCE_PUSH_BEFORE_TRAIN.json](results/SOURCE_PUSH_BEFORE_TRAIN.json) match [ADMISSION.json](results/ADMISSION.json) and the released files. Scientific source remained byte unchanged during formal training, native recovery and Outer evaluation. Later commits added aggregate-only rendering, packaging, public verification and disclosure helpers, not model/selection changes.

Actual admission passed six critical CPU tests; all ten F0/C2 validation replays had maximum score error **0**, with all 201 original threshold decisions exact. Sixteen fresh configurations passed three discarded engineering backward steps on a complete real FIT patient; these were pre-training admission, not discarded formal epochs. Feature masks, shared initialization, route switches, mismatch support, branch slices, parameter counts and remaining gradient paths were checked.

The unchanged original preprocessing, patient roles, class weights, loss, AdamW settings, epoch budget, threshold rule and full-patient updates were used. FP32, TF32 off, deterministic cuDNN, 4,096-window chunks and the original patient/seizure pooling were preserved. No new extraction, Raw copy, manual feature, sampling or hyperparameter search occurred.

Before formal fitting, a read-only diagnostic initially encountered duplicate OpenMP initialization when importing torch first. Diagnostic import order was corrected to the existing scientific module's numpy/scipy/sklearn/torch order; no unsafe duplicate-runtime override was used. Initial SSH transport/signature failures were resolved through a strict pre-trusted host-key-pinned transport, without publishing credentials or weakening host identity checks. These were engineering access/admission issues, not evidence of changed data or failed credentials.

## Actual training and native recovery

[TRAIN_CELLS.csv](results/TRAIN_CELLS.csv) records every selected epoch, threshold and checkpoint/validation digest. [TRAIN_HISTORY.csv](results/TRAIN_HISTORY.csv) records **997 completed epochs across 80 fresh fits**. Ten reference cells are explicitly marked reused and assigned zero new epochs, rather than counted as retrained.

Run tag 02 suffered a Windows native **c0000005** access violation during atomic PyTorch LAST serialization for `DROP_SPECTRAL_ENTROPY`, fold 4. Windows Application Error and the native stack identify the failure site; the responsible native module/root cause was not established. It is not attributed to a proven driver or scientific error.

Before resuming, the following were verified and preserved:

- All **63** completed fresh cells' checkpoint, validation and completion hashes.
- The last complete epoch **9**, selected best epoch **6**, 18 model tensors, 17 Adam slots and Python/numpy/torch/CUDA RNG state.
- LAST SHA256 **8badab449db61aaad4f271786e0895c87130dbe3ad228d5e52fc1f2d6c5b9bd3** and a whole-file-verified private backup.
- The failed zero-byte temporary save and original logs, without deleting or overwriting them.

The same frozen command resumed in a fresh process with new log tag **03**, skipped completed cells and resumed incomplete epoch **10** from the exact epoch-9 state. No model, data, loss, preprocessing, optimizer, threshold, selection rule, library or scientific source change was made. All remaining fits completed. See [NATIVE_RESUME_AUDIT.json](results/NATIVE_RESUME_AUDIT.json). This run is disclosed as recovered, not failure-free.

## Freeze, sealed Outer and verification

| Gate | Actual verified result |
| --- | --- |
| Training | 80 fresh fits + 10 exact reuses = 90 cells |
| Internal verification | All 90 checkpoint/validation replays; maximum error 0; all 201 decisions exact |
| Pre-Outer freeze | `ALL90_FROZEN_BEFORE_NEW_OUTER` |
| Outer sealing | `ALL90_OUTER_CELLS_SEALED`, before metric reporting |
| Actual patient endpoints | 1,440 rows × 9 metrics = 12,960 independent checks |
| Resampling | 324 nominal intervals + 18 Macro-F1 family intervals checked independently |
| Maximum interval discrepancy | 5.551115123125783e-16 |
| Final verification | `COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER` and independent `PASS` |

Selected-model freeze digest: **da15eb1db029cf7510e8f462c21b758c8bdd7c9e3f7a939eae6142793ea74a0a**.

Full pre-Outer freeze digest: **2bee5ce863af8774491b6e53f58cf0fd5775d562cf3644e107809604abdf15c9**.

All-90 Outer score-seal digest: **6dc73306f04ffefa9ffee471470786b3e190d16a304a81b8be5b9b2225a1f033**.

Each of the fifteen new Raw-consuming arms retained all **1,471,965 valid windows** over 80 patients and 7,635 channels. Physiology-only deliberately consumed zero Raw inference windows. The FiLM mismatch control changed all 1,471,965 eligible windows; valid singleton windows were zero. Historical F0/C2 sealed predictions were reused byte-for-byte after hash verification.

The production driver exited normally with:

```text
F0_COMPLETE18_FIVE_FOLD_OUTER_VERIFIED_COMPLETE
```

The independent private endpoint/interval verifier printed:

```text
F0_ALL18_FULL80_INDEPENDENT_VERIFICATION_PASS
```

## Public export and visible QA

Only an explicit public allowlist was exported: anonymous aggregate CSV/JSON/Markdown and aggregate-only SVG/PDF/PNG. The package contains **35 entries**, is **345,074 bytes**, and has SHA256 **32f9cbadde7916695fbc5480f3c12401026c2ce389fde933aa8f848a55a658e6**. The transfer hash was independently checked, then every public file hash was checked against [PACKAGE_MANIFEST.json](results/PACKAGE_MANIFEST.json). The ZIP itself is not committed.

The public verifier passed with:

```text
F0_18_ARM_PUBLIC_RESULTS_REPORT_AND_MANIFEST_PASS
```

Both actual PNG previews were visually inspected; see [FIGURE_QA.md](FIGURE_QA.md). The immutable export's renderer manifest correctly retains its original generation-stage status; the separate visible QA record completes that step without altering the server-exported files or their digests.

No raw waveform, identity, private per-patient/channel prediction, weight, scaler, cache, permutation mapping, log or credential is included. New runtime artifacts and graphics caches stayed on the server's D drive; existing original trusted sources were read in place. This study is exploratory repeated-test, not independent confirmation. No new arm was selected or modified from its Outer results.

The release additionally supplies `verify_git_release.py`: it compares the pushed branch tip with HEAD and checks exact committed Git blob digests against all public-export, scientific-source and renderer manifests. Scoped Git attributes preserve immutable exported bytes, including standard Windows line endings and Matplotlib SVG path spacing; reports are not silently normalized after hashing.
