# Unit-consistent FeaturePR actual execution audit

## Source and unchanged scope

Scientific source was pushed andremote tip verified before formal fitting at`0e7be02746f73700d4d8b22d365b47a10fa687f4`. Source archive SHA256:`43dd0cac9d48158d56691b9cd27b5cf8149a582cf947f94ecb9e8b073ad6253f`. All14deployed top-level files matched the push. The12scientific/execution files are bound by[ADMISSION.json](results/ADMISSION.json)andthe actual[SOURCE_PUSH_BEFORE_TRAIN.json](results/SOURCE_PUSH_BEFORE_TRAIN.json);README,this audit andlater public-only release helpers are not training dependencies.

Parent Raw dual-view source/freeze/score seal andall historical results remain unchanged. This is a new derived physiology bank in a new Ddrive namespace,not an overwrite of previous feature banks,model inputs,checkpoints orthresholds. The original NoFeaturePR input/scaler remains bit unchanged. There is no new raw extraction,raw copy,window sampling,feature addition,feature deletion,architecture change,loss change ordynamicRawanchor update.

## Actual admission

-Four unique changed-path preprocessing tests passed locally andon the server:unit-rescaling/legacy nondegenerate parity,explicit degenerate/missing handling,TRAIN/query isolation,andreproduction of the old mixed-unit discontinuity. No toy accuracy is reported.
-All5real derived banks preserved original roles/patient/channel/label/index arrays,legacy inputs andNoPRmatrices. TRAIN-only units exactly matched the old raw global TRAINscales/variances. Units/scalers are never fitted on CALorASSESS. All6original physiological features andtheold inactiveVariance pair are retained in the same14wide layout.
-All45reference model cells matched their original initial state,preprocessor,CALlogits,archive indices,selected epoch/threshold andcheckpoint hashes. These are3fixed references,not better-looking post-outcome substitutes.
-All30existing initialTRAIN Rawanchors matched initial-state/TRAINpatient/index/window bindings. They are reused,not falsely reported as new fits;the pre/no-pre anchors remain fixed. Independent full Rawanchor replay is still required before new assessment.
-A genuine completeTRAINpatient NoFeaturePR update matched the old unmodified procedure's logits,loss,allgradients andoneAdam model/state exactly,max gradient error0. Admission updates were discarded. This verifies that the control is not weakened by changed inputs orneural math;complete30cell retraining parity is still checked after actual training.
-All6fresh paths passed finite complete-patient backward admission,including nonzero CNN/bothRawview/physiology input-block gradients. No checkpoint,epoch,threshold ornew assessment result was used for admission tuning.

## Actual input-scale evidence

See[PREPROCESSING_ADMISSION.csv](results/PREPROCESSING_ADMISSION.csv). TRAIN/CALactive input maxima are now10.41–12.69across all5folds. In fold3CAL,the old4,048,492,800maximum became10.422847747802734;in fold5CAL,the old15,553,996,800became11.898587226867676. No clipping orselective patient/coordinate exclusion was used. The NoFeaturePR matrices remained bit identical. All14raw TRAINunit variances were positive in these actual banks;the unsupported zero-variance-unit fallback was not needed.

Numerical correction is established for the admitted banks. It does not prove a Macro-F1/AUROC improvement,a biological explanation orfinal goal success. No new outcome has been filled from these admission statistics.

## Production state and remaining evidence

The foreground coordinator launched around08:49:11Asia/Shanghai;admission passed andformal training began around08:50:41on2026-10-20. At08:52:08,the positively identified coordinator42464/launcher26308/worker42528 was active,7/135cells had completed(4fresh/3reference),andtrainingstderr was empty. The admissionstderr contains the normal4-test PASS transcript. Preserve the sustained SSH owner;an observation timeout orstate file alone never authorizes a restart.

Production uses the existing isolatedPyTorch2.11.0+cu128 runtime,FP32/TF32off,ownworkerCPU16-31 andOMP/MKL/OpenBLASthreads1. All90fresh fits and45reference reuses must complete. Then require5independent physiology bank replays,30Rawanchor replays,30NoFeaturePR/bothPR fresh-control identities,135CALreplays/selections/preprocessors frozen,and135scores sealed before unified metrics. Independent verification must cover1080patient-fold-seed rows/9720metrics,162nominal/14additional-family intervals andalloriginal role/window coverage.

The old mixed-unit comparison is fully reported as an additional diagnostic;mathematical correctness does not promise an all-metric gain against that procedure. The fixed main performance andretained-module rules remain mandatory. No control promotion,post-assessment threshold rescue,gate edits orautomatic newOuter. The full objective still requires actual final80-patient Macro-F1>0.64,improvedAUROC andstable retained-module contributions. The historical mainOuter0.653828is unchanged,not a guaranteed repaired-model result.

Only source,hashes andanonymous aggregate audits are public. Raw,identities,individual predictions,weights,preprocessors,anchors,caches andruntime logs stay on the trusted server.
