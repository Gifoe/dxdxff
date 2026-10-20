# Actual execution and scope

The8diagnostic sources were pushed before execution at`c2b5fe922347a2adc22a11ea6db360ed99048323`. Ten deployed top-level files matched the push exactly. Source archive SHA256:`eb714db710594eaf1375743b4b2c06bf91b818fa98a56f5dbd392448dbfbb398`.

Five unique critical-path tests passed locally andon the server. The foreground owner started around2026-10-20 08:16:30Asia/Shanghai. All45private diagnostic caches completed before aggregate reporting;cache seal SHA256:`cc5ab7f0100c1ec08c81fe9d257a089bc21215ca335b739fa1bf06f45999088e`. The source-bound parent120-model freeze andprediction seal were only hash-verified;parent assessment private scores/metrics were not opened by this diagnostic.

All30new complete TRAIN/CAL forward bodies reproduced stored CALlogits exactly and preserved selected model state. B0's15storedCALcells were reused under their already verified hashes,not falsely described as new forwards. No model fit,epoch selection,deployment threshold change orASSESS/Development/Outer prediction took place. Inherited bank/cache deserialization contains other metadata oralready storedlabels,butonly legalTRAIN/CALarrays enter this diagnostic,andonlyCALlabels enter threshold sensitivity.

Independent verification passed45selectedCALcells,360leave-oneCALthreshold/metric checks,60latent phase summaries and75unlabeled score summaries. MaximumCALmetric discrepancy0. The model epoch had already been selected using all8CALpatients;holding one out only for threshold selection does not make this unbiasedCV. The reported table averages diagnostic cells equally andis not the active goal's patient-average endpoint.

The owner ended exit0. Audit andverifierstderr were empty;teststderr contained the normal5-test PASS transcript. No worker was restarted,no source/checkpoint/preprocessor/cache or existing result was replaced. The11-member anonymous export was18910bytes,SHA256:`824030d5d2ec50f04cb1df3488c93c55c1f1b821588a2bd968e41b9d428f1448`. All10payload hashes andthe standalone public checker passed before repository installation. No private arrays,identities,weights,raw/cache orlogs were fetched.

Actual CALMacro-F1 versus leave-oneCALthreshold sensitivity wasB0:0.649321to0.631468,PRUNED_REF:0.634867to0.606413,FULL:0.624001to0.598332. These are selection-conditioned diagnostics,not new validation orOuter results. FULL's initial-anchor standardizedTRAINcoordinate-meanRMS was0.672341versus0.281459forPRUNED_REF;coordinate-SDRMS1.526341versus1.089073. This is not proof of causal normalization failure or proof that anchor refresh would help.

An additional observed issue changed the next action:fold3CALshared15.765%probability saturation andmillion-scale logits across all3bodies. Rather than immediately adding a new normalization/model orrescuing thresholds,a separately frozen unlabeled input arithmetic extension inspected the existing TRAIN/CALpreprocessing andheadinput scales. Its source/protocol andoutputs are separate from the immutable8-source/11-member diagnostic. No model,patient,coordinate orthreshold was changed ordiscarded by this inspection. The active goal remains unachieved.

## Separately frozen numeric extension:actual execution

The2extension sources were pushed before execution at`47250a4dea32e6a70f6c74968cf55a42b04ed518`andhash-checked on the server. The original8diagnostic sources remained unchanged. A foreground isolatedPython process used the same owned CPU affinity andsingle-thread environment;no background orphan or unrelated process was touched. It completed exit0 on the first attempt.

Two unique math/mask selftests passed. All20FeaturePR/noFeaturePR TRAIN/CALinput reconstructions matched existingFP32banks exactly. All280coordinate rows and60head-path rows completed;120NumPy-versus-torchCPUfloat64 projection checks passed,max relative discrepancy5.613945828101816e-15. No label,neural update,newRaw forward,classification metric orthreshold fit was used. Four anonymous export members matched the3payload hashes;archive23664bytes,SHA256:`d34ff924f4b26104249340c14fb2ea1254812c5724d8f53b49343527c6067637`.

The old internal CALnumerical explosion is reproducible from the existing frozen input pipeline,not fabricated by a new scorer. Its effect on model selection andfinal performance has not been causally isolated. OriginalOuterMacro-F1=0.653828andall source/data/model/threshold records are unchanged. A new unit-consistent PR experiment,if pursued,must be separately preregistered andfairly retrained;no post-hoc threshold rescue orautomatic newOuter follows from this audit.
