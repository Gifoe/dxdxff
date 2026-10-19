# Branch-wise nonlinear PR residual: six-feature DREChannNet

Implementation/admission pending; **no new candidate result yet**. The prospective [design](DESIGN.md) was pushed before the preceding PR-factorial's new assessment scores were calculated. Its motivation includes an outcome-free raw-summary scale confound, not a selected classification outcome. The final lock is not executable until the completed dependency seal is filled and verified.

This package tests one fixed candidate. It preserves the original single-head path at zero initialization and adds128bounded residual gain parameters on branch-wise nonlinear comparisons. The same128parameters exist in all five new variants; no second classifier is introduced. No loss, feature, training budget, seed or architecture search is performed.

| Arm | Original summary PR | New nonlinear residual context | Purpose |
| --- | --- | --- | --- |
| B0 | both | no added residual | exact original internal baseline,15 verified reuses |
| FULL | both | patient Z for both branches | one fixed candidate |
| WO_FEATURE_PR | raw only | channel-wise LayerNorm for physiology; patient Z for raw | remove all physiology-branch PR, keep branch/scale/capacity |
| WO_RAW_PR | physiology only | patient Z for physiology; channel-wise LayerNorm for raw | remove all raw-branch PR; TRAIN-only initial raw affine scale |
| WO_BOTH_PR | neither | channel-wise LayerNorm for both | joint removal, not zero inputs |
| CHANNEL_CONTEXT | both | channel-wise LayerNorm for both | capacity-matched control of NEW patient context |

All75new fits use original matched initializers, seed42/43/44 and all five held-out paths. TRAIN34–36/CAL8/ASSESS8 remain inside each original FIT. The33unique assessment patients are **not** the80-person Outer. All90selected model cells freeze and all90score cells seal before metrics. Report all nine endpoints,108nominal intervals,8additional primary-family intervals, source/fold/seed/pooled/coverage tables. No best-seed or best-Outer selection.

Raw PR off no longer feeds a raw branch roughly200times smaller solely because of the deleted standardization. Its global affine preconditioner is fitted only on all TRAIN channels under the matched initial encoder and frozen. It is not a current-encoder population reference or exact distribution matching during optimization. Per-channel residual normalization never compares another channel. All15anchors must independently replay exactly before assessment.

Use sustained foreground SSH to run `run_all.ps1 -SourceCommit <verified40hex> -RunTag <newtag>` after source push. All code/runtime/checkpoints stay on D, trusted caches are reused without copying, and the admitted isolated runtime/owned-worker affinity is retained. Failures stop for diagnosis rather than blind retry.

The fixed qualification rule can admit only a separately frozen core-module study. It does not authorize new Outer or establish the active goal. Final Macro-F1>0.64, AUROC improvement, retained-core-module contributions and uncertainty remain actual-execution requirements, not presumed successes. All results on this repeatedly viewed cohort are exploratory.
