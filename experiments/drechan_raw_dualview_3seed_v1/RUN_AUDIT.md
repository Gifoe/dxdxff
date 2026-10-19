# Raw dual-view execution audit

## Status and evidence boundary

Formal production training started at2026-10-20 07:30:56 Asia/Shanghai,after corrected admission passed. This is a running internal study,not a completed performance result. The target remains final original-protocol patient-average Macro-F1>0.64 with improved AUROC and supported retained components. The historical original80-patient Outer Macro-F1=0.6538278218104221 is unchanged. No candidate internal score may replace that number or be directly subtracted from it.

There are90fresh fits and30exact parent reference reuses:8arms x3seeds x5original within-FIT paths=120cells. New variants each have15473parameters; references have11425(B0)and9329(PRUNED_REF). All120selections must replay/freeze and all120scores seal before unified metrics. Internal assessment has33unique patients,40fold appearances and3seeds; repeated fold appearances are first averaged within patient/seed,then across seeds,then patients equally. It is exploratory repeated-test,not independent confirmation or a new80-patientOuter.

## Source pushes before fitting

- Initial fixed study pushed at`f8d43f1f2095d010424b6089ffd3cf2294089ed6`.
- Initial source archive SHA256:`8149a03c95918102ff9ebc57ab05e2e13ff07feb8a8a39149ffa59758844e5fc`.
- Measured admission repair pushed and remote tip verified at`ffb3d386b271ac39be13413c74b22afd41ce9c9c`,before formal training.
- The corrected[ADMISSION.json](results/ADMISSION.json)binds all12scientific/execution file hashes. They remain fixed during formal fitting. README,this audit and the public-only checker are not production training dependencies.
- Parent source,freeze and sealed reference predictions are bound by[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json). The completed parent study and all earlier results remain byte unchanged.

## Pre-training admission failure and diagnosis

The first admission attempt stopped before formal fits on an exact gradient comparison. Independent and production forward logits/loss were bit-identical,but the independent auditor shared one float64 cast between the two Raw views. Production cast the two branches separately. These forward-equivalent formulas have distinct finite-precision backward graphs: the shared branch accumulates gradients before one cast,whereas production casts each branch gradient before accumulation.

The actual full-TRAIN-patient[precision probe](results/ADMISSION_PRECISION_PROBE.json)used125channels/36580valid windows. The shared-cast auditor differed in4parameter gradient tensors,max absolute difference1.1874362826347351e-08. The separately cast independent auditor matched logits,loss,allgradients and oneAdam model/optimizer update exactly,max error0. Probe updates were discarded; there were no formal fits,new assessment outcomes or newOuter.

Reproduction provenance:the diagnostic helper`admission_precision_probe.py`was actually executed with the original`f8d43f1...`versions of the12dependencies listed in its JSON. It imports the old admission comparator. The helper snapshot itself is SHA256`8c0f91bf47755a58b58430f80ec3e24387ed9499a582a3478670d8c39ac6e658`and is published with the repair commit. It must not be run against the repaired auditor to claim reproduction of the old failure. It also refuses execution once formal private state exists. The old source archive,old auditor/test copies and original failed-attempt logs remain preserved on the trusted server.

## Minimal engineering repair and preservation

Only`admit.py`and`test_model.py`changed among the12bound files before formal fitting. The independent formula now preserves production's separate cast branches. A sixth unit test verifies forward and input-gradient equality. The production`model.py`,`view_math.py`,loss,data,roles,threshold rules,optimizer,budget and numerical tolerance did not change.

All30pre-admission initialTRAIN anchors were preserved in a separate recoverable namespace. Admission regenerated the30anchors under the corrected source binding. The actual[anchor equivalence audit](results/ANCHOR_REPAIR_EQUIVALENCE.json)verified every old/new mean,scale,initial model identity,TRAIN identity/count and pre/no-pre kind:all30numeric tensors identical,max absolute difference0. Only the auditor/test hashes differ in their source bindings. No completed formal epoch existed to discard or replay.

## Corrected admission and production

-6critical unit tests passed.
-30reference cells passed exact initialization/checkpoint/preprocessor/role/CAL bridges; see[ADMISSION_REUSE.csv](results/ADMISSION_REUSE.csv).
-30initialTRAIN-only pre/no-pre anchors were fitted without CAL,assessment orOuter labels; see[ANCHOR_ADMISSION.csv](results/ANCHOR_ADMISSION.csv).
-Full real-patient production versus independent logits,loss,allgradients and oneAdam update were bit-identical,max gradient error0.
-All6fresh paths passed finite full-patient backward admission,including nonzero Raw-CNN and both Raw-input block gradients.
-The admission JSON's`new_formal_fits=0`is a pre-training snapshot,not a claim that the later production run has not started.

Production uses the existing isolatedPyTorch2.11.0+cu128 runtime,FP32/TF32off,own-workerCPU16-31 affinity and oneOpenMP/MKL/OpenBLAS thread. It reads existing raw caches in place and does not copy raw data. One foreground SSH owner holds the sequential coordinator; detached orphan training is forbidden on this server. Complete atomic LAST saves model,Adam,RNG and history. A genuine failure must be diagnosed before exact resume with new log names; no duplicate jobs or unrelated process changes.

## Release checks still required

Completion requires120selected/replayed/frozen cells,30independent anchor replays,120sealed score cells,960patient-fold-seed rows/8640metric values,144nominal intervals and12fixed additional-family intervals. The independent private verifier must recompute all patient metrics and bootstrap intervals. A standalone public checker can verify hashes,anonymous table structure,training/reuse counts,aggregate identities and fixed gate calculations,but cannot recompute private patient bootstrap draws. The qualification rule must not be relaxed from outcomes; no better-looking control is promoted after the fact. Internal completion or admission PASS is not active-goal completion and does not authorize newOuter.

Only source,hashes and anonymous aggregates are public. Raw waveforms,identities,individual predictions,weights,anchors,preprocessors,caches and runtime logs remain on the trusted server.
