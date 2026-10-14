# Stage A arithmetic and execution repairs

No extraction bands, windows, cohort, gates, tolerance or clinical-label rule was changed.

The initial bulk float32 reconstruction missed the archived E14 parity gate in 3 of 106,890 coordinates (2 RMS, 1 variance). The original generator performs the within-record mean separately for each channel on its contiguous full 44-dimensional evidence rows, followed by 1-dimensional float32 mean/population-standard-deviation reductions for each coordinate across seizures. Replaying that exact arithmetic yielded a bit-identical archived E14: maximum difference 0, zero failed coordinates. `E14_EXACT_AGGREGATION_DIAGNOSTIC.json` retains the actual diagnostic.

The correction reuses the pinned original evidence helper for both engineered and Raw measurements. Its full 13-feature/44-coordinate intermediate layout is retained solely to reproduce the original float32 arithmetic; only the prespecified seven DELTA families and their 14 final coordinates enter E1/D1. This is not an expanded predictor.

Original initial source SHA256:

- construct_delta.py: d0dcd4a7f3e78ee60dcc98de9eadea324a35b6f638ea4058268a375aa2ef9140
- raw_physiological_features.py: 3ed07fbe61073ceab6e79a42a7e77ba409c45b3aa84050942c8798a2cf706471
- prepare.py: 5f8479023644d6d3b29813a6638f27416478565ea65b916dff50a4c96c536774

An exact-arithmetic full run was then interrupted at patient 32 by Windows native NumPy `_multiarray_umath.cp313-win_amd64.pyd` access violation 0xc0000005. It produced no completed gate or classification result. The initial runtime and interrupted run are retained privately.

The engineering repair adds atomic per-patient reconstruction files bound to source/code/protocol hashes, with checksum verification before reuse. Mathematical-library thread counts are limited to one; no algorithm or tolerance is substituted. A new runtime is used because changing code must not rewrite an existing input freeze. Only completed, identical-binding patients may be resumed. Source and checkpoints are not deleted. A failed hidden-process launch with empty logs is not counted as an executed reconstruction.

The protocol lock remains byte-identical. The final corrected gate and code/test hashes are reported separately; initial failure and its diagnostic are not hidden.
