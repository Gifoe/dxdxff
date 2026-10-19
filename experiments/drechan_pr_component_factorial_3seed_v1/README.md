# Six-feature DREChannNet: separate Feature PR / Raw PR factorial

Status: admission passed,45 fresh fits running; **no new assessment results yet**. The12 server source files matched pre-training GitHub commit `589c79695aa2eea0152a7ca98865a368af855cca` byte-for-byte. See [execution audit](RUN_AUDIT.md).

This is a diagnostic step toward the active goal, not a replacement for an improved final model. The existing main model retains its original Outer Macro-F1 **0.653828**. The completed temporal-contrast experiment failed its prespecified mechanism gate; its sources, gates and conclusions are not edited or reopened here.

The current six-feature main model's previous ablation removed both PR operations together. That cannot identify which operation changes ranking and which mainly changes calibration. This new fixed 2x2 experiment measures their separate conditional contributions and interaction, with identical capacity, original training/threshold rules and three seeds. No claim that all retained modules must help is assumed.

| Arm | Summary Feature PR | Final Raw PR | New internal fits |
| --- | --- | --- | ---: |
| FULL | on | on | 0; verify/reuse15 exact internal B0 cells |
| WO_FEATURE_PR | off | on | 15 |
| WO_RAW_PR | on | off | 15 |
| WO_BOTH_PR | off | off | 15 |

Within-FIT TRAIN/CAL/ASSESS are the already frozen assignments (34–36/8/8 patients per original fold). Their33 unique assessment patients are **not the80-patient Outer**. All three seeds42/43/44 are reported. Repeated use of this internal research assessment and all historical Outer exposure is disclosed.

Feature PR off uses its own TRAIN-only imputer/scaler on the original summary without patient Z. Raw PR off bypasses only the original final96D patient Z. Neither deletion removes a branch, changes input width/initializer, skips pre-reference, or changes window conditioning. The network remains trainable; no trained full weights initialize ablations. There is deliberately no post-result normalization repair; scale sensitivity is a limitation to be measured, not concealed.

## Execution

Server source/runtime/weights stay on D, using the existing isolated runtime and trusted caches in place. After source push, run `run_all.ps1 -SourceCommit <verified40hex> -RunTag <newtag>` in a sustained foreground SSH session. It holds an exclusive coordinator lock, performs admission,45 fresh fits/15 exact reuses, selection verification, all-cell score sealing, reporting and independent verification. Every atomic LAST has optimizer and RNG; a failed process stops for diagnosis rather than blind retry. The original completed experiment is read-only.

All nine endpoints,108 nominal paired/absolute intervals, four additional primary-family intervals, folds/seeds/source/pooled/coverage tables and reuse audits are required. No new Outer is run by this protocol. Stage completion must not be described as achieving the active goal; improvements and core-module evidence remain to be established in subsequent predeclared experiments.

Only code and anonymous aggregate artifacts are public. No identifiers, individual predictions, weights, preprocessor arrays or runtime logs are exported.
