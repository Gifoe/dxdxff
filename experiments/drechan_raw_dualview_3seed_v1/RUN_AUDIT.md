# Raw dual-view execution audit

## Status and evidence boundary

Formal production training started at2026-10-20 07:30:56 Asia/Shanghai,after corrected admission passed. The internal study is now complete and independently verified;the terminal evidence is appended below and the fixed qualification failed. The target remains final original-protocol patient-average Macro-F1>0.64 with improved AUROC and supported retained components. The historical original80-patient Outer Macro-F1=0.6538278218104221 is unchanged. No candidate internal score may replace that number or be directly subtracted from it.

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

## Release requirements and verification boundary

Completion requires120selected/replayed/frozen cells,30independent anchor replays,120sealed score cells,960patient-fold-seed rows/8640metric values,144nominal intervals and12fixed additional-family intervals. The independent private verifier must recompute all patient metrics and bootstrap intervals. A standalone public checker can verify hashes,anonymous table structure,training/reuse counts,aggregate identities and fixed gate calculations,but cannot recompute private patient bootstrap draws. The qualification rule must not be relaxed from outcomes; no better-looking control is promoted after the fact. Internal completion or admission PASS is not active-goal completion and does not authorize newOuter.

Only source,hashes and anonymous aggregates are public. Raw waveforms,identities,individual predictions,weights,anchors,preprocessors,caches and runtime logs remain on the trusted server.

## Verified terminal,2026-10-20

-Production finished90fresh model cells/1091complete epochs plus30exact reference reuses/374historical epochs. Formal training completed around07:56:56Asia/Shanghai. No formal process was restarted,completed epoch discarded,scientific file changed or configuration dropped. The pre-training admission repair above remains the only repair;all12corrected scientific/execution files stayed byte unchanged throughout formal fitting.
-All120selected models/preprocessors/CALoutputs/epochs/thresholds replayed exactly. All30initialTRAIN pre/no-pre anchors independently replayed on complete original TRAIN support. Pre-assessment freeze SHA256:`3bbaaf51d159d6dcf4f61d57633e4d70fc653ad2b9a216a68e2b27a95899188c`.
-All120score cells sealed before unified metrics,seal SHA256:`ea236252f9cb7e6335323a213fcea36aff1557ddd828d82a21b488f565690a9e`. No newOuter inference occurred. New assessment outcomes were not read before freeze/seal or used to edit the model,thresholds or gate.
-Independent verification passed960patient-fold-seed rows/8640endpoint values,144nominal and12fixed additional-family intervals,all role assignments and actual raw-window coverage. Maximum interval discrepancy3.3306690738754696e-16. Training,selection,assessment and metric-verifierstderr were empty;admissionstderr was the normal6-test unittestPASS transcript. The foreground owner ended exit0 with`RDV_INTERNAL_STAGE_VERIFIED_NO_NEW_OUTER_ACTIVE_GOAL_NOT_YET_ACHIEVED`and`RDV_FOREGROUND_OWNER_TERMINAL`.
-The explicit31-member anonymous export is116278bytes,SHA256:`fb13d05f374f5d4505a2f2217b1660a1e7eb8ee0b041178f923eabc9a3fcf2e1`. All30payload hashes and the public-only standalone verifier passed before installation into the repository. Previously published admission payloads matched byte-for-byte and the two extra diagnostic audits were preserved. No private prediction,raw,checkpoint,anchor,preprocessor,cache orlog was fetched.
-The candidate's internal AUROC improved0.009712andAP improved0.013641versusB0,but Macro-F1 declined0.019647andEZ-F1 declined0.028479. All3seeds had positive AUROC and negative Macro-F1 differences. FeaturePR's two endpoint family intervals were positive;RawPR's AUROC interval still crossed0. The global-view contrast was an AUROC/Macro-F1 tradeoff,not a supported two-metric gain. The fixed performance,expanded-reference and global-view conditions failed;qualification stayedFAIL.
-No control was promoted,no threshold or selection rule was tuned after assessment,no newOuter was authorized and no original main-model result was replaced. OriginalOuter Macro-F1 remains0.6538278218104221. This completed internal study does not achieve the active objective. The observed ranking/decision tradeoff motivates further legal TRAIN/CAL-only investigation,not a post-hoc threshold fitted to these assessment labels.
