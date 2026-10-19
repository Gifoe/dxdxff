# Temporal Patient-Relative Contrast — DREChanNet, three seeds

Status: real-data admission **PASS**, before formal training. All8CPU tests, five exact initializer/old legal-validation replay bridges and real complete-patient gradients for all3new reference paths passed. See [admission](results/ADMISSION.json) and [role counts](results/INTERNAL_ROLE_ADMISSION.csv). No new accuracy numbers are supplied by this source-only document.

The reference is the six-feature, no-Variance/no-explicit-head-interaction single-head model from `075305f1aa372a9ac34faa19e689d2cca96fe7bd`. Its historical Outer patient Macro-F1/AUROC/AP/EZ-F1 were 0.653828/0.752255/0.585240/0.467424. These are historical exploratory references, not independent targets for tuning.

## Fixed four-arm experiment

| Arm | Window-level reference | Purpose |
| --- | --- | --- |
| B0 | No new module; exact prior model | Matched reference |
| B1 | Other valid channels, same patient/seizure/time; leave-one-out median/MAD | Temporal local-context contrast |
| B2 | All TRAIN valid seizure/channel rows at the same time; current CNN | Capacity-matched global-reference control |
| B3 | Same patient's reference at a wrong time with identical full channel support | Correct-time diagnostic |

The three new arms share the same 1,776-parameter residual fusion of original Raw24 pre-reference residual `r` and relative evidence `q`. The final layer starts at zero, giving exactly the B0 initial function; no trained F0 weights initialize new fits. Original early FiLM, late Gate, two summary PR operations, one64-hidden head and losses remain unchanged. The six retained measurements are beta, low gamma, partial high gamma80–125Hz, RMS, line length and entropy10–125Hz. Variance remains masked everywhere.

Exact leave-one-out medians/MAD use four sorts rather than a quadratic channel-pair tensor. Background statistics are explicitly stop-gradient in all three arms; recipient and pooling paths and the existing final96D patient Z retain gradients. Minimum3other valid channels, epsilon1e-4, MAD floor1e-6 and q clipping[-8,8] are fixed before any outcomes. Ineligible reference coordinates use q0 but retain original `r`; no patient/window is dropped.

B2 is recomputed before every optimizer update and before calibration/evaluation using current CNN weights. Its weight-independent normalized inputs may be held on GPU, but no signals are copied into a new disk cache. B3 never borrows another held-out patient's inputs: it applies a fixed nonzero time shift within identical full channel-mask support groups of the same seizure, retaining and reporting singleton groups.

## Roles, selection and stopping

- Seeds42/43/44; original80patients,7,635channels,256seizures,1,471,965valid Raw250 windows and original five folds.
- First stage: new TRAIN/CAL/ASSESS roles entirely within each original FIT, label-free source-proportional hashed allocation. CAL and ASSESS each receive ceil(15%ofFIT). TRAIN alone fits preprocessing, conditioning scale and global reference; CAL selects epochs/thresholds. Original Development/ownOuter excluded.
- All60internal models are fresh:4arms×3seeds×5folds. Old full-FIT B0 cannot be reused because it trained on the new ASSESS subjects. Original AdamW/learning rate/weight decay/clipping/loss/max30/min6/patience6/FP32/chunk4096 remain fixed.
- All60selected cells and calibration replays freeze before ASSESS; all60score cells seal before unified endpoints. Exact checkpoints/optimizer/RNG and hash-bound patient score caches support engineering-only recovery.
- User's tolerance is simultaneous B1-B0 point differences≥−0.01 on AUROC/AP/Macro-F1/EZ-F1, on mean-over-seed estimates and in at least2/3individual seeds. This is not formal noninferiority.
- Also require positive B1-B2 mean AUROC andAP and both positive in at least2/3seeds, with mean Macro/EZ loss no worse than1pp. B3 is diagnostic, not a selector. No gate relaxation or replacement search after seeing results.
- Only if both gates pass: original-FIT final training of all4arms×3seeds×5folds, with original Development selecting epoch/threshold only. Five exact B0seed42cells may be reused after state/validation proof;55other fits are fresh. All60final selections freeze before newOuter; all60scores seal before metrics. Otherwise stop and publish the complete negative internal experiment without newOuter.

Primary aggregation averages repeated fold appearances within each unique patient, then metrics across three seeds, then equally across patients. It is not a prediction ensemble or best-seed result. Report nine endpoints, per-seed/fold SD, sources with small-source suppression, pooled endpoints separately and coverage.10,000seed42patient-paired bootstrap draws produce63nominal intervals and12additional primary-family intervals using approximate Bonferroni percentile tails. Bootstrap is conditional and does not capture training or historical-selection uncertainty.

## Execution

Code and runtime reside on server D; trusted old caches are read in place. All local source bytes must be transferred/verified, real admission must pass, and this branch must be pushed and remote-tip-verified **before** formal training.

```powershell
# Run via a sustained foreground remote owner; do not detach then disconnect.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\drechan_temporal_pr_contrast_3seed_v1\run_all.ps1 -RunTag admission01 -StopAfter Admission
powershell -NoProfile -ExecutionPolicy Bypass -File D:\drechan_temporal_pr_contrast_3seed_v1\run_all.ps1 -RunTag production01 -SourceCommit <verified40hex>
```

The driver holds an exclusive coordinator lock, preserves log names, stops on worker failures and never blindly retries. `admit.py` requires CPU mathematical/model-path tests, five exact B0 initializer/old legal-validation replays, strict internal role isolation and real complete-patient gradient checks. `verify.py` independently replays selected references/calibration and recalculates metrics, gates, intervals, expected score indices and Raw coverage. `package_public.py` exports only a strict anonymous allowlist.

## Evidence limitations

Historical `raw_latent_contrast_optimization_seed42_v1` PatientZ AUROC+0.028655 with[0.010792,0.049047] is factorial mean/min/max over8frozen-CNN probe backgrounds, **not a patient-paired95% confidence interval**. Weak prior relation/affine/ranking results remain negative evidence. No claim of novelty, propagation, guaranteed PR benefit or universal stability is made.

This experiment is exploratory/repeated-test because the cohort and original reference were already viewed. New within-FIT roles do not create fresh patients or erase that history. Only source, hashes and anonymous aggregates may be public. Signals, identifiers, per-patient/channel results, weights, global-reference tensors, preprocessors, caches and logs remain on the trusted server.
