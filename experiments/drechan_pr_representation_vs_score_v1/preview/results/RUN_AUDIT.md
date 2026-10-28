# Early two-model outer audit

The user requested all available outer results except TimeConv-CNN on
2026-10-28. This is the expressly requested early disclosure of the complete
SVM/DREChanNet subset, not completion of the three-model experiment.

- Main frozen scientific source: `eede217c80985bbb4f91f8df5580d4fcfbef0f6b`.
- Early-preview source and amendment: `985d745` (before any new outer scoring).
- Main production03 training remains unchanged, with one GPU worker.
- Preview uses CPU FP32, with CUDA hidden in its separate inference process.
- All 70 selections were snapshotted and hash-frozen before preview scoring.
- All 70 score files sealed before new endpoint computation.
- Coverage: both models, all seven conditions, all five folds, 80 unique patients
  and 7,635 channels for each condition. No patient or condition was discarded.
- Historical NoPR/FullPR test outputs remain the exact hash-bound originals.
- Score transformations use native NoPR logits; ranking uses the exact frozen
  NoPR anchor. AUROC/AP/MRR/Top-1 match NoPR for every patient.
- The 20 validation forward cells actually needed by CPU inference replayed
  within the fixed tolerances, with identical threshold classifications.
- The first preview admission additionally replayed unused FullPR on CPU and
  stopped on its fold-2 numeric tolerance before any new test scoring. Its
  source, freeze and logs remain private and preserved. Version 2 reads FullPR
  only from its original sealed predictions, without changing tolerances for
  the CPU forward paths used. See the protocol amendment.

Independent verification passed 1,120 patient-condition rows and 10,080 endpoint
values, all 270 nominal intervals, and four available primary family intervals.
The family size remains six, including the two pending TimeConv comparisons.
The paired within-source patient bootstrap retains all four source sizes
36/21/15/8 and uses 10,000 draws. Count-matrix estimates were checked against
independent direct-index resampling and independent endpoint formulas.

Only the eight explicitly listed anonymous payloads and manifest were exported.
Whole archive SHA256:
`8a5dc8507ef9ce06c6377e2b4877773a6e67520eef9621f85ac5fc36ee3d6175`.
The local package check ran before allowlist extraction. A separate local check
reconstructed overall patient-weighted means, equal-source means and worst
sources from all 56 source rows and confirmed historical NoPR/FullPR values
against the existing published aggregate CSVs. All checks passed.

No signals, identities, individual endpoints, predictions, weights, scalers,
reference parameters, embeddings or runtime logs were downloaded. Original JBHI
tables and figures remain unchanged. The later three-model GPU report is still
pending and must compare its numerical results with this CPU preview.
