# Raw waveform reference-scale view fusion

Target: patient-equal Macro-F1>0.65; actual complete-cohort Raw-only reference
0.630493. Prior classifier-only and warm learned-dispersion experiments were
negative. This is a separate information-retention hypothesis, not a renamed
positive result or an expansion of those frozen budgets.

Completed and independently verified:30 internal plus10 final matrix fits.
Full80-patient Macro-F1 Native/control/Precompand/selected is
**0.630493/0.628211/0.631432/0.629086**. Precompand AUROC/AP improved to
0.759384/0.570215,but its Macro difference interval crosseszero. **The0.65goal
was not reached.** See [actual results and limits](RUN_REPORT.md).

The original waveform view removes each window's amplitude by its own SD.
The second view uses a common pre-only scale across complete channels within
each seizure, then fixed asinh companding. It preserves relative channel and
pre/post amplitude differences within the clipping range. Both actual
waveforms pass through the **same frozen CNN**, and only their learned neural
phase summaries are fused by a576parameter,zero-initialized bounded matrix.
There is one unchanged classifier, no added RMS/variance/bandpower scalar input
and no independent D1 branch. The preprocessing scale is not called a learned
physiological feature or a provenance repair.

A same576parameter fusion of the original view controls capacity. Complete
patient channels and genuine windows remain unchanged; phase/seizure pooling
and patient Z are fully differentiable through the fusion matrix. Frozen-body
phase caches accelerate exact matrix fitting, not sampled-window training or
an end-to-end CNN update. No generic residual/multi-view novelty is claimed.

The protocol fixes30 internal fits across15 original ownFIT paths, with CAL-
only epoch/threshold selection and HELD-only ownFIT candidate selection. All
765patient-encoder cells, Native replay and full actual-patient raw/cache
loss/gradient gates must pass first. No original Development/Outer is used in
the internal stage. No performance values are filled before verified runs.

```powershell
& D:\raw_precompand_macro_seed42_v1\run_internal.ps1 -RunTag 01 -StopAfter Admission
& D:\raw_precompand_macro_seed42_v1\run_internal.ps1 -RunTag 02 -StopAfter All
& D:\raw_precompand_macro_seed42_v1\run_final.ps1 -RunTag 01 -StopAfter Admission
& D:\raw_precompand_macro_seed42_v1\run_final.ps1 -RunTag 03 -StopAfter All
& E:\Anaconda\python.exe D:\raw_precompand_macro_seed42_v1\verify_final.py
& E:\Anaconda\python.exe D:\raw_precompand_macro_seed42_v1\verify_raw_coverage.py
```

Use a sustained foreground SSH owner, new RunTags and preserved private logs.
Code and anonymous aggregates are pushed; raw, IDs, labels, feature caches,
weights, scores and logs stay private. Internal completion is not the0.65goal.
A later fixed fivefold evaluation must freeze both matrices/body/thresholds
and all four named procedures first, and remain exploratory repeated-test.
