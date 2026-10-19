# Pruned DREChannNet with parameter-matched branch fusion

Prospective implementation; no new result yet. See [fixed design](DESIGN.md) and [executable protocol](PROTOCOL_LOCK.json). The original mainOuter Macro-F1 remains0.653828; this internal stage is not a replacement score.

Seven configurations,three seeds42/43/44,all five original within-FIT paths: B0,JOINT_PRUNED,FULL,WO_FEATURE_PR,WO_RAW_PR,WO_BOTH_PR,WO_PRE_REFERENCE.90fresh fits plus15exact internalB0reuses. Both unsupported guidance operations and128unsupported new residual gains are absent in all new variants. The original rawCNN,pooling,pre-reference,summaryPRs and six physiological features remain; FULL changes only activation fusion relative to its9329parameter JOINT_PRUNED control. One64-hidden head and one scalar classifier, no new patient-context layer.

All105selections freeze and all105scores seal before unified reporting. Require840patient-fold-seed rows/7560values/135nominal/10additional independent interval checks. Full details and fixed qualification in the protocol. A point qualification is not proof of the active goal and does not authorize newOuter. Do not replaceFULL with a better-looking control or relax the rule.

Server code/runtime and weights stay on Ddrive. Existing raw caches are read in place. Push source before formal fits, run admission then production under one sustained foreground SSH owner; diagnose genuine failures before exact atomic resume with a fresh log tag. No detached orphan, duplicate training, changed data/precision/budgets or unrelated GPU/system changes.

All internal assessment is exploratory repeated-test. Only source,hashes and anonymous aggregates may be published; patient data,predictions,weights,preprocessing,raw/cache andlogs remain private.
