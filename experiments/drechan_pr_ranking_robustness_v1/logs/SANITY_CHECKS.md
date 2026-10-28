# Executed verification and provenance audit

Completed 2026-10-29, Asia/Shanghai. All raw signals, identities, individual
predictions/endpoints, weights, scalers, caches and runtime logs remain private.

- A-C: 45 score cells froze and sealed; 720 patient-condition rows independently
  checked. All 80 patients and 7,635 channels retained. AUROC/AP use independent
  tie-aware formulas; MRR/Hit use the canonical channel tie order. Score-Z rank
  endpoints, including Hit@3/5, match NoPR exactly for every patient.
- 480 individual paired-difference rows were written in trusted private storage.
- Source-weighted means reconstruct overall means; four equal source means and
  metric-wise minima reconstruct CM and Worst. Source groups are not LOSO.
- Fixed-budget hit, recall, precision, NEZ count, lift and actual channel counts
  were verified. All six coverage curves start at zero, end at one and are
  monotone. The 5/10/20% curve points match the budget tables.
- D: 57 eligible patients, 217 seizures; HUP27/LZU15/multicenter9/pediatric6.
  Their 5,863 original channels yielded 5,735 common valid channels; 128 were
  excluded by the same label-independent rule, with no eligible patient lost.
- All 80 original 14D feature summaries reconstructed bit-for-bit from the
  existing frozen per-seizure window features. No extraction was changed.
- Every D neural cache reconstructed the direct All representation and logit
  bit-for-bit. Subset pooling and PR references use only selected seizures.
  K=1 population SD is zero. Preprocessors and validation thresholds stayed fixed.
- All K=1/K=2 combinations were enumerated (maximum eight seizures); the same
  subsets were shared across models. No probability ensembles were formed.
- All 342 D patient-model cells sealed before endpoints. Independently verified
  3,666 repetition rows, 1,026 patient-K rows and 144 paired intervals; all eight
  endpoint coverages are 57 patients. Another 1,026 individual paired rows remain
  in trusted private storage.
- Bootstrap uses patients within sources, 10,000 paired draws; count-matrix and
  independent direct-index implementations agree. Prespecified families:6,12,9,36.
  The SVM AUROC contrast passes the existing six-test Score-Z family but crosses
  zero in the separate twelve-test NoPR family. Both results are retained.
- The seven-condition experiment also passed all 105 cells/1,680 patient rows/
  15,120 endpoints, 405 nominal and six adjusted intervals. Its final GPU output
  matches all 10,080 endpoint values of the early two-model CPU preview exactly.

## Execution deviations retained in the audit

The user authorized a 70-cell early SVM/DREChanNet disclosure during the
seven-condition run. This replaced the original common disclosure timing; the
remaining TimeConv source, budgets and selection rules stayed frozen. Final
public status explicitly scopes the 105-cell freeze/score seal to the primary
pipeline. The private seals were not rewritten and original JBHI figures/tables
were not overwritten.

D ran as CPU FP32 frozen inference while the separate GPU training continued.
Additional CPU partitions reused the same immutable input plan and scientific
inference function, changing only partition counts and completed-cache reuse.
Finished files were atomically promoted without replacing a different result;
duplicate completion was checked for exact array equality. One auxiliary worker
was intentionally stopped when free virtual memory reached about5.9GiB; it
released memory to about9.7GiB. Its patients remained in the original complete
schedule. All342 canonical cells subsequently completed and verified.

SSH/Tailscale transport became unreachable after the20:05 successful status.
Existing processes were not relaunched. After connection recovery, both verified
terminal records and complete archives were read, and no task Python workers
remained. Transport reset was not treated as a training failure.

Intermediate status text was finalized only after independent PASS. Original
archives were retained on the server. CSV result bytes, scientific sources and
private freeze/score seals were unchanged by metadata finalization. Public
allowlists and whole-archive SHA were verified before local extraction.

Final anonymous archives:

- A-D: `539dd61e587a845b33754f3a83bf6bcac5d975f5464ad110340b0db7cfb7d2fa`.
- Seven conditions: `b2831887e688df71c0d5395da2feea3f45c488371756e510c61ff9f2a253aa6e`.
- Earlier A-C snapshot: `04aad3aa1a95afd1d1dcee4ec8f33f784e00bc6eb7e1a6a34ded4eb4a5f1af36`.

Figures A-C were rendered and visually inspected; PDFs retain vector elements
and embedded editable text, with SVG counterparts and600-dpi PNG previews.
