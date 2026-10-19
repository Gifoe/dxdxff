# Execution audit

## Frozen source and live admission

- Pre-training source/lock pushed to GitHub: `589c79695aa2eea0152a7ca98865a368af855cca`; remote branch tip verified before launch.
- All12 server files, including README, matched the local pushed source SHA256 before the coordinator started on2026-10-20 around05:00 Asia/Shanghai. The11-file scientific/execution binding is stored in `ADMISSION.json`; README/status prose is not part of that binding.
- Runtime uses the previously admitted isolated Python3.13.9/Torch2.11.0+cu128/NumPy2.3.5 environment, with owned-worker affinity restricted to logicalCPUs16–31 and OMP/MKL/OpenBLAS threads1. No system configuration or unrelated process was changed. The previous system-wide native-fault root cause remains unproven; this is not a claim that the hardware/environment has been repaired.
- Four unique changed-path unit tests passed: shared initialization/Variance masking, exact Raw PR bypass, finite full-network gradients in all four arms, and TRAIN-only no-Feature-PR imputation/scaling.
- All15 old internal B0 cells passed exact initializer, role, preprocessor, selected-CAL-logit, epoch and threshold reuse checks. These are not original full-FIT Outer models; they have exactly the new protocol's internal TRAIN/CAL/ASSESS roles.
- Each of the three newly enabled deletion paths completed a genuine full-largest-TRAIN-patient forward/backward with finite gradients. These admission checks made no optimizer update and used no ASSESS or Outer outcomes.
- Production started after admission. First new `WO_FEATURE_PR`, seed42/fold1 cell completed8epochs. Logs, LAST/BEST checkpoints, RNG/Adam state and private calibration outputs remain on the server. No new assessment metrics have been read at this milestone.

## Resume and publication rules

Keep the foreground SSH stream alive: this server has previously reaped detached child jobs at short-lived SSH exit. Coordinator holds an exclusive lock. Do not launch a second worker or reset checkpoints. A failed worker exits the coordinator for diagnosis; a genuine engineering resume must use a new RunTag with original source/input/checkpoint bindings intact.

The current stage is complete only after all60 selections, score seals and the independent480-row/4320-value/108-nominal/4-family verification pass. That status still does **not** complete the active model-improvement goal: no new architecture or final Outer model is produced by this diagnostic alone. Publish all signed effects, including unfavorable and zero-crossing contrasts. Do not relabel an internal Macro-F1 as the original0.653828 Outer result.
