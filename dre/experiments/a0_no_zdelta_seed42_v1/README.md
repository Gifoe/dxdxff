# A0 without ZDELTA — seed42

Completed source-free A0 ablation, keeping Patient Z. No raw EEG extraction.

Matched outer (80 patients): AUROC **0.736977** versus A0 **0.739768**; AP **0.530653** versus **0.537887**; Macro-F1 **0.624236** versus **0.624919**. No evidence of improved outer discrimination. This is not the previous source-conditioned D2 w/o ZDELTA experiment.

- [Final report](FINAL_REPORT.md): full development/outer metrics, intervals, interpretation and caveats.
- [Protocol](PROTOCOL_LOCK.json): seed, original training/selection rules, named zero mask and predeclared parity limits.
- [Outer primary metrics](results/OUTER_SUMMARY.csv) and [paired bootstrap](results/OUTER_PAIRED_BOOTSTRAP.csv).
- [Development metrics](results/DEVELOPMENT_SUMMARY.csv), [unique-ID secondary](results/DEVELOPMENT_UNIQUE_ID_SUMMARY.csv) and [bootstrap](results/DEVELOPMENT_PAIRED_BOOTSTRAP.csv).
- [Mask](results/FEATURE_MASK_AUDIT.csv): 26 ZDELTA columns zeroed, 62 retained, 88D topology unchanged.
- [Model selection](results/MODEL_SELECTION.csv): five independent validation thresholds per arm.
- [Pretrain audit](results/PRETRAIN_AUDIT.json), [freeze](results/FREEZE_AUDIT.json), [independent verification](results/INDEPENDENT_VERIFICATION.json) and [reporting-only checks](results/POSTHOC_VERIFICATION.json).

## Execution

Requires the trusted Windows server's original frozen feature banks, original model/trainer sources and sealed A0 predictions. Private dependencies are intentionally not distributed. Do not substitute raw caches or another D2 model.

Run `code/baseline_admission.py`, then `code/run.py` with phases in order: `prepare`, `train`, `development`, `freeze_outer`, `outer`, `verify`. Run `verification/posthoc.py` last. See argument defaults for private runtime/source roots; source hashes and unchanged science are checked before proceeding. Complete evaluations refuse a fresh scientific rerun. Training uses the unchanged original trainer's exact checkpoint resume; outer resume only reuses matching hash-sealed score files.

`verification/posthoc.py` was added after outcomes only for mask/input and pooled aggregate verification. It cannot train, select thresholds or generate model predictions. The frozen training/evaluation code was not changed after its first run.

All result tables are anonymous aggregates. No checkpoints, signals, banks, identifiers, per-patient predictions, runtime logs or credentials are published. The incomplete prior Patient-Z-removal draft is not part of this branch's submitted experiment. Results are exploratory repeated development/outer, not blind confirmation.
