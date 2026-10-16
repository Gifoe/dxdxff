# Fresh Raw-only factorial: final verified results

## Outcome

**The full-cohort patient Macro-F1 >0.65 target was not achieved.**
The prespecified ownFIT-selected procedure is **0.624365**, versus unchanged
Native Raw-only **0.630493**: paired difference **-0.006128**, nominal 95% CI
**[-0.024435, 0.011413]**. The highest fixed secondary point is PRE_MEAN
**0.634936**, not >0.65. Its difference from Native is **+0.004443**, CI
**[-0.016732, 0.025174]**. It is not adopted as the primary after viewing Outer.

These are actual complete fivefold results, not internal validation, pooled
substitutes, an oracle or unrun estimates. Historical Outer has been viewed
repeatedly: **exploratory repeated-test, not independent confirmation**.

## All seven full80 procedures

Each metric is calculated within a patient and averaged equally over the same
80 patients / 7,635 canonical channels / 256 seizures. Macro-F1 averages the
EZ and NEZ class F1 values. All nine metrics are finite for all80 in every row.
Channel-pooled metrics are separate in `OUTER_POOLED.csv`, not the target.

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native reference | 0.630493 | 0.741296 | 0.550114 | 0.426475 | 0.834511 | 0.757014 | 0.784454 | 0.700000 | -0.041381 |
| Frozen precompand reference | 0.631432 | 0.759384 | 0.570215 | 0.420221 | 0.842644 | 0.766018 | 0.802328 | 0.725000 | -0.066138 |
| WINDOW_MEAN, fresh | 0.615596 | 0.737468 | 0.542611 | 0.394279 | 0.836912 | 0.757389 | 0.786666 | 0.700000 | -0.068719 |
| WINDOW_LOGENERGY, fresh | 0.619238 | 0.741587 | 0.539783 | 0.395209 | 0.843266 | 0.766505 | 0.757319 | 0.637500 | -0.085079 |
| PRE_MEAN, fresh | 0.634936 | 0.755925 | 0.567903 | 0.437536 | 0.832337 | 0.760687 | 0.762320 | 0.662500 | -0.035605 |
| PRE_LOGENERGY, fresh | 0.619795 | 0.746326 | 0.558110 | 0.409963 | 0.829626 | 0.753827 | 0.788943 | 0.700000 | -0.038115 |
| SELECTED, primary | 0.624365 | 0.752879 | 0.563181 | 0.421839 | 0.826891 | 0.749666 | 0.786467 | 0.700000 | -0.027580 |

Exact [patient means](results/final_stage_v1/OUTER_PATIENT_MEAN.csv) and
[all absolute intervals](results/final_stage_v1/OUTER_ABSOLUTE_CI.csv).
The primary absolute Macro-F1 CI is [0.595035, 0.654231]; its upper bound
above0.65 does not mean the actual point estimate reached the target.

## Matched attribution

All four fresh networks share 12,529 parameter positions, identical initial
CNN/head tensors within eachfold, legal data, loss, optimizer, threshold grid
and maximum/early-stop budget. The historical references have different
training histories; comparison with them alone cannot isolate a new module.

| Patient Macro-F1 contrast | Difference | Nominal paired 95% CI |
|---|---:|---|
| WINDOW_LOGENERGY - WINDOW_MEAN | +0.003642 | [-0.010834, 0.018623] |
| PRE_LOGENERGY - PRE_MEAN | -0.015142 | [-0.027867, -0.002161] |
| PRE_MEAN - WINDOW_MEAN | +0.019341 | [0.000437, 0.038402] |
| PRE_LOGENERGY - WINDOW_LOGENERGY | +0.000557 | [-0.023418, 0.024510] |
| Pooling-by-normalization interaction | -0.018784 | [-0.038287, 0.000731] |

Pre-only common-scale normalization under mean pooling retains a positive
matched point/nominal interval, but does not establish an improvement over the
strong historical Native reference. Energy pooling is not supported as the
explanation for better Macro-F1; under PRE it is worse than matched mean
pooling in this repeated test. The interaction interval includes zero.
No novelty or clinical mechanism is claimed for generic log-energy pooling.

The completed internal study had42freshfits/1,062epochs and did not establish
a general energy-versus-mean Macro advantage. Its legally frozen fold1–5
choices were FROZEN_PRE, NATIVE, PRE_LOGENERGY, PRE_LOGENERGY, PRE_LOGENERGY.
These were not changed to PRE_MEAN after seeing Outer. Internal appearances
and this full80 cohort remain separate: [INTERNAL_REPORT.md](INTERNAL_REPORT.md).

## Fold and source stability

| Procedure | Fivefold Macro-F1 sample SD | HUP, n=36 | LZU, n=21 | Multicenter, n=15 | Pediatric, n=8 |
|---|---:|---:|---:|---:|---:|
| Native | 0.035327 | 0.651040 | 0.551194 | 0.698488 | 0.618699 |
| Frozen precompand | 0.037602 | 0.665682 | 0.548155 | 0.686961 | 0.591797 |
| WINDOW_MEAN | 0.030433 | 0.658184 | 0.521263 | 0.649117 | 0.608718 |
| WINDOW_LOGENERGY | 0.020913 | 0.657591 | 0.532967 | 0.639338 | 0.635420 |
| PRE_MEAN | 0.033930 | 0.681064 | 0.579489 | 0.643545 | 0.556769 |
| PRE_LOGENERGY | 0.034842 | 0.660948 | 0.553879 | 0.641067 | 0.567748 |
| SELECTED | 0.042909 | 0.659158 | 0.547428 | 0.680819 | 0.563907 |

No source-uniform improvement is demonstrated. PRE_MEAN improves HUP/LZU
points but is below Native in multicenter/pediatric groups. Lower fold SD
does not rescue a worse mean. One seed and small source groups limit stability
claims. Full nine-metric [source](results/final_stage_v1/OUTER_SOURCE.csv),
[fold](results/final_stage_v1/OUTER_FOLDS.csv) and
[SD](results/final_stage_v1/OUTER_FOLD_SD.csv) tables are included.

Intervals use10,000seed42 patient-resampling draws of fixed fitted-model
outcomes, not retraining uncertainty. They are nominal, not corrected for all
repeated comparisons, and cannot restore independent-test status.

## Raw-only model and actual time support

Four fresh candidates receive only Raw250 waveform, valid mask and original
times: oneCNN/onehead, no handcrafted physiological predictor coordinates,
D1 outputs, prior trained embeddings or physiological classifier. PRE's fixed
pre-only common RMS scale normalizes the waveform; it is not appended as a
predictor. LOGENERGY pools learned CNN responses, not measured raw bandpower.

Of12,529 parameter positions,4,096 are incident on the preserved64literal-zero
historical head slots;8,433 participate in the fresh forward function. This is
not a claim that every counted weight is necessary. Frozen precompand is the
earlier two-normalization Raw reference with shared CNN, residual mixing and
one frozen classifier, not a newly fitted single-view candidate. SELECTED is
a fold-specific procedure, not one globally uniform new architecture.
[MODEL_SPEC.md](MODEL_SPEC.md) provides the exact data flow.

An independent post-seal, label-free full80 Raw census measured:

| Support category | Valid windows |
|---|---:|
| Pre-onset reference, time <0 | 724,672 |
| First phase, 0 to <8 seconds | 199,960 |
| Second phase, 8 to <20 seconds | 299,940 |
| Time >=20 seconds | 247,393 |
| Total forwarded through each fresh encoder | 1,471,965 |

The fixed pre-reference/two-phase aggregator uses **1,224,572** windows. The
247,393 late windows are encoded but are **not** claimed to influence final
classification. No time support or mask was changed. The24,995available
channel-seizure units all have genuine pre support; no canonical channel lacks
either post-onset phase across all its seizures.

There are18,632invalid padding slots, including2,584nonfinite stored times.
All valid waveform samples and valid times are finite. Invalid padding is
never converted into valid data. Stored valid time coordinates range about
-29.001 to29.001seconds. These are actual census results, not inferences from
cache sample rate or default preprocessing arguments. They do not establish
low-frequency acquisition provenance or recover >125Hz content from Raw250.
[Census](results/final_stage_v1/RAW_SUPPORT_AUDIT.json).

## Four requested components: prior actual evidence

Earlier matched frozen-encoder/readout deletions measured Macro-F1 differences
of -0.050031 without pre-latent reference, -0.009622 without phases, -0.002415
without seizure SD, and -0.092557 without patient-wise Z. The SD interval
crossed zero. These are conditional readout findings, not four end-to-end
necessity proofs or universal physiological claims. Their full scope and
negative warm-adaptation experiments remain in the
[mechanism ledger](../raw_temporal_context_macro_seed42_v1/MECHANISM_EVIDENCE.md).
The current factorial retains all four ingredients and does not turn these
earlier findings into evidence that >0.65 has been reached.

## Execution and independent checks

Final execution actually fit20freshmodels over**474epochs**. Epoch wall time
totaled**1,740.717493seconds**, mean**3.672400s**, excluding orchestration,
admission, inference, replay and reporting. Training ran about07:02–07:32
Asia/Shanghai,2026-10-17, without trainer restart or completed-epoch loss.
All35model/threshold cells were frozen, then all35prediction cells sealed
before opening the complete Outer truth archive. Fixed reference predictions
are exact reused outcomes, not secretly refitted models.

- All20Development replays have maximum logit error0 and identical decisions
  at every one of201candidate thresholds. Allfour random initial states match
  within eachfold; all fresh selected checkpoints are actual epochs>=1.
- All560patient rows and nine metrics were independently recomputed. Maximum
  metric difference:**1.1102230246251565e-16**.
- All162absolute/paired/factorial CI metric rows, fold means/sampleSD, source
  aggregates and coverage were independently checked. Maximum bootstrap
  difference:**2.220446049250313e-16**.
- A local empty-stderr SSH pump timed out, but the same healthy trainer and
  foreground SSH owner continued; this did not trigger a training restart.
- The supplemental census first aborted at import because torch-before-NumPy
  initialized duplicate OpenMP runtimes. It read no Raw data. Matching the
  verified program's NumPy-first order passed the changed-path import test.
  No unsafe duplicate-library override was used.
- Its next attempt over-rejected nonfinite times in invalid padding. A full80
  diagnostic proved zero nonfinite valid samples/times. The census assertion
  was corrected to the existing valid-mask contract and reports the padding.
  Earlier sources/logs are preserved; no model, data, mask, score, threshold,
  truth set or scientific source binding changed.
- Final census and statistics checks passed and perform no training or new
  prediction. `run_audits.ps1 -RunTag 03` is the final deployed audit invocation;
  earlier log tags remain private on the server.

The legacy reader physically unpickles mixed historical metadata banks but
returns no labels in label-free mode. It is not claimed to provide physical
isolation from every historical file. Only legal training labels drive fits;
the prediction function receives no labels or feature-bank measurements.
Original FIT/Development/Outer identities remain disjoint within each fold.

Key SHA256 locks:

- OwnFIT choices: `8c1cc33a6ae4644bd4e82acc926e16ff1aed371ed6c6d4502c79a84f7d017cc6`
- Final input: `7d9da8759f090f83a5d44b2c3c74ab95e58cbdd7bf3b87341ac4fb3751ee0dca`
- Models/thresholds: `73b7708f4a6f41e87337fb03797398f3e99eb54913fd7339dd63843718685ef8`
- All35scores: `0e8c2454ac21ad3af3b980294da28e06f01327ef26844a09210babeb4a2245ae`
- 17-file public ZIP: `b0e3d825a53b812ea01baad15cde0762ae00cdad244ffba32d7cd9a0ef7ac826`

Final scientific sources/protocol were committed before original Development
access (`76f54f3`); later repairs were audit/report-only. Raw, identities,
labels, weights, individual predictions, private caches and runtime logs stay
on the trusted server. Only code, hashes and anonymous aggregates are public.
Local checks: `python audit_completion.py`, then
`python verify_public.py --write-manifest` and `python verify_public.py`.
The completion audit passes delivery consistency but explicitly records
`target_point_achieved=false`. The broader >0.65 goal remains unachieved.
