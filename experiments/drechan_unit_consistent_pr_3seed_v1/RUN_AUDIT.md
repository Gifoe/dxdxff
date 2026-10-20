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

## Subsequent live observation and public-only checks

At09:02:00Asia/Shanghai on2026-10-20,the same coordinator42464/launcher26308/worker42528were confirmed alive.61/135cells were complete:40fresh fits and21references. The current cell wasWO_BOTH_PR/seed42/fold3,epoch8;trainingstderr remained zero bytes. These are execution counts only,not newassessment metrics.

A short-lived observation connection encountered an SSH-banner timeout. The original sustained SSHowner continued emitting live coordinator heartbeats throughout;the subsequent observation connected normally andconfirmed the same process identities. No training restart,checkpoint rollback orscientific change was made in response to an observation failure.

Added`verify_public.py`andits six-test standard-library regression suite outside the12bound sources. Actual admission-only verification andall6tests passed locally. Checks coversource bytes,45reference admissions,30reused anchor admissions,10TRAIN/CALpreprocessing rows,andnegative cases for tampered source digests,duplicate aggregates andmissing final manifest. The pre-fit admission's`new_formal_fits=0`remains a historical snapshot;it is not used to deny the active formal training. Full public verification remains pending actual completion anddoes not claim to regenerate private patient rankings/bootstrap draws.

## Actual training completion and selection freeze

At09:17:25Asia/Shanghai on2026-10-20,the actual remote`TRAIN_STATUS.json`reported`ALL135_COMPLETE`,90fresh fits/1052new epochs and45exact reference cells/571historical epochs. The coordinator had moved from training to independent selection replay;trainingstderr remained zero bytes. Nothing was restarted ortruncated. A separate09:04:32source-byte check had confirmed all12bound server files still exactly matched their pre-fit admission hashes.

At09:19:46,the actual remote freeze status was`ALL135_FROZEN`,SHA256`17dd41550f82f170207251c609e2a4d231b21e3d84a583f3b48c1f8dce95dcb2`. The selection worker completed every cell,including the required independent5physiology-bank/30Rawanchor replays and30fresh-null-control identities;assessment was then positively identified running under coordinator42464. Its stderr was empty at that observation. No score was read to alter the model,threshold orprotocol.

The sustained,previously authenticated foreground owner then observed the coordinator advance through`verify_metrics`and`package_public`,print`UCP_INTERNAL_STAGE_VERIFIED_NO_NEW_OUTER_ACTIVE_GOAL_NOT_YET_ACHIEVED`,andexit0around09:21Asia/Shanghai. This proves command-level termination under the fixed coordinator's gates,not local inspection of the final metric/manifest files. The public package SHA,payload bytes,independent-verification JSON andactual performance table must still be retrieved andchecked before any final numerical claim orrelease.

## Trusted retrieval interruption,not a scientific retry

New observation connections began failing at SSHhandshake signature verification around09:20. The original sustained authenticated owner remained healthy until its normal completion. Subsequent Paramiko andnative WindowsOpenSSH probes both failed the Ed25519handshake signature. Native strict-host-key diagnostics matched the pre-existing pinned host fingerprint butreported`incorrect signature`;the Curve25519andECDH-NISTP256exchange paths both failed. A group14-SHA256probe was not offered by the endpoint,andthe old network route timed out. Native diagnostic probes deliberately disabled allauthentication methods,so those failures occurred before password authentication;they do not establish invalid account credentials. The underlying host/service/link cause is not proven.

No host key was accepted anew,no signature verification was disabled,no known-host record orcredentials were changed,andno server/service/unrelated job was restarted. No scientific process was retried. The final aggregate package has not yet been fetched,andno predictive numbers,package hashes orqualification outcome are manufactured from the successful exit marker. Only the earlier actual admission release is locally available;its public-only check still passes,butfull-result verification remains pending. The active research objective is not achieved by this command-level internal completion.

## Retrieval restored and actual release verified

The preceding interruption describes the09:20diagnostic state,not the final access state. On2026-10-20the user confirmed that the same endpoint was accessible andexplicitly requested the original password connection. Comparing the two local transport paths exposed the omitted difference:the existing authorized password helper disablesEd25519,butthe later pinned adapter re-enabled andpreferred that algorithm. Outcome-free,unauthenticated handshake tests found valid ECDSAandRSA-SHA2signatures,whileEd25519failed. Thus the earlier broad inference that SSHitself was unavailable was incomplete. A different exchange algorithm while still forcingEd25519did not test this host-key-algorithm difference.

At14:53:37Asia/Shanghai,the original authorized password entry successfully returned the expected server hostname andtime. It was used without altering the helper,server configuration,credentials,global known-host file,scientific code ordata. Its pre-existing first-use host-key policy was used at the user's explicit direction;no independent out-of-band fingerprint confirmation is claimed. Handshake signature verification was not disabled. No scientific command was rerun:only completed compact status/export files were read.

The actual status files confirm90fresh fits/1052epochs plus45references/571historical epochs,all135frozen selections andall135scores sealed before unified metrics. Freeze SHA256:`17dd41550f82f170207251c609e2a4d231b21e3d84a583f3b48c1f8dce95dcb2`;score-seal SHA256:`7e6d68fb25e85a4649421b6ef8a708a8810d3b4917d5a37ab05cbfe2ebf5c3a6`.

The actual131113-byte anonymous export SHA256is`4393623381be6459a683a449b4afca5fd0afd7b42647e5bd08632685402928b6`. All34unique safe archive members and33payload hashes were checked before extraction. Previously published admission bytes matched exactly andwere not overwritten. Full standalone public verification andall6helper regressions passed after transfer. The server's independent verifier reports1080patient-fold-seed rows/9720metric checks,162nominal/14additional-family intervals,maximum interval discrepancy3.3306690738754696e-16andalloriginalrole/window coverage PASS. The12bound scientific sources remain byte unchanged.

The actual fixed qualification isFAIL. FULL internal Macro-F1=0.6083336001755916andAUROC=0.7534740689518008;FeaturePRhas positive Macro-F1/AUROC evidence,butRawPRdoes not have a positive meanAUROC contribution,andthe globalview lowers Macro-F1. No gate,threshold,patient,seed orcontrol was changed after these outcomes. No better-looking control is promoted,no newOuter is run,andthe final research objective remains unachieved. The original80-patientOuter0.6538278218104221is a separate unchanged historical result.
