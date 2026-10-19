# Execution audit

## Frozen source and live admission

- Pre-training source/lock pushed to GitHub: `589c79695aa2eea0152a7ca98865a368af855cca`; remote branch tip verified before launch.
- All12 server files, including README, matched the local pushed source SHA256 before the coordinator started on2026-10-20 around05:00 Asia/Shanghai. The11-file scientific/execution binding is stored in `ADMISSION.json`; README/status prose is not part of that binding.
- Runtime uses the previously admitted isolated Python3.13.9/Torch2.11.0+cu128/NumPy2.3.5 environment, with owned-worker affinity restricted to logicalCPUs16–31 and OMP/MKL/OpenBLAS threads1. No system configuration or unrelated process was changed. The previous system-wide native-fault root cause remains unproven; this is not a claim that the hardware/environment has been repaired.
- Four unique changed-path unit tests passed: shared initialization/Variance masking, exact Raw PR bypass, finite full-network gradients in all four arms, and TRAIN-only no-Feature-PR imputation/scaling.
- All15 old internal B0 cells passed exact initializer, role, preprocessor, selected-CAL-logit, epoch and threshold reuse checks. These are not original full-FIT Outer models; they have exactly the new protocol's internal TRAIN/CAL/ASSESS roles.
- Each of the three newly enabled deletion paths completed a genuine full-largest-TRAIN-patient forward/backward with finite gradients. These admission checks made no optimizer update and used no ASSESS or Outer outcomes.
- Production started after admission. First new `WO_FEATURE_PR`, seed42/fold1 cell completed8epochs. Logs, LAST/BEST checkpoints, RNG/Adam state and private calibration outputs remain on the server. No new assessment metrics have been read at this milestone.

## Verified terminal

- All45fresh fits completed621full epochs;15exact B0 reuses represent168historical internal epochs. No completed epoch was discarded, no training error was observed, and all selections replayed before score freezing.
- All60selected cells frozen under SHA256`a6572de2f6d455a695b0d7d8c71e8bc42cad2ead9094766baf56d8ed689f0b58`; all60scores sealed under`7ddd3d7c7e616411eaa5762e8e61cc85ec8512af92ad480f94589cb593205fac`.
- Server independent verification passed480patient-fold-seed rows,4320endpoint values,108nominal and4additional intervals. Maximum interval discrepancy3.33e-16. All role indices and actual raw-window coverage verified. No new Outer was run.
- The sustained coordinator session finished exit0 with `PRF_DIAGNOSTIC_STAGE_VERIFIED_NO_NEW_OUTER_ACTIVE_GOAL_NOT_YET_ACHIEVED`. New SSH observations temporarily timed out reading the protocol banner, before authentication. TCP22 and an actual OpenSSH banner were verified; the same pinned transport then reconnected. No experiment was restarted or credential changed.
- Anonymous28-member export SHA256`c3903fab6d7d9e1675a01790b62d51e10f6015a5252f7cb074ee2c5d836cd3e5` matched its manifest and all27payload hashes before extraction. The public-only release checker additionally checks public means/SD/contrast points/evidence flags and source hashes; it does not pretend to recompute patient-bootstrap distributions without private rows.
- The next branch-residual design was publicly committed at`119677e32ff8e185719bc709e97a5ee88798c645` while this factorial was still training, before its new assessment scoring. Its approximately197-fold initialization RMS scale diagnosis comes from the real unlabeled input admission, not from selecting a classification outcome.

## Resume and publication rules

Keep the foreground SSH stream alive: this server has previously reaped detached child jobs at short-lived SSH exit. Coordinator holds an exclusive lock. Do not launch a second worker or reset checkpoints. A failed worker exits the coordinator for diagnosis; a genuine engineering resume must use a new RunTag with original source/input/checkpoint bindings intact.

The current stage is complete only after all60 selections, score seals and the independent480-row/4320-value/108-nominal/4-family verification pass. That status still does **not** complete the active model-improvement goal: no new architecture or final Outer model is produced by this diagnostic alone. Publish all signed effects, including unfavorable and zero-crossing contrasts. Do not relabel an internal Macro-F1 as the original0.653828 Outer result.
