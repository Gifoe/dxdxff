# Phase3 — Small decision-targeted mechanisms

Registered after Phase2, not retrospectively substituted into earlier trials. The best previous morphology correction had Macro-F1+1.708263 points versus D1, short of the explicitly confirmed+2-point goal, despite AUROC+2.702943/AP+4.269884/EZF1+2.818638 points and improvement in5/5 folds. Phase3 fixes only two candidates, ten new training cells:

- M7: exactlytwo added scalar parameters condition correction amplitude on uncalibrated frozen-D1 logit magnitude; same BCE/model otherwise.
- M8: same morphology correction topology/parameter count, fixed.75 weightedBCE+.25 patient SoftMacroF1 to test objective alignment. This is a training mechanism, not an invented architecture gain.

```powershell
$env:D1_PHASE1_CODE='D:\d1_small_module_exploration_seed42_v1\code'
$env:D1_PHASE2_CODE='D:\d1_small_module_exploration_seed42_v1\phase2\code'
$env:DGM_COMPLETED_CODE='D:\dgm_net_seed42_v1\code'
$env:DGM_BASE_CODE='E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code'
& 'E:\Anaconda\python.exe' -m pytest D:\d1_small_module_exploration_seed42_v1\phase3\tests -q
& 'E:\Anaconda\python.exe' D:\d1_small_module_exploration_seed42_v1\phase3\code\run.py --runtime D:\d1_small_module_exploration_seed42_v1_phase3_runtime
```

All shared science/data/seed/schedule/selection/threshold rules stay frozen; only the declared mechanism changes. Phase3 controls and all eight candidates are reported, regardless of failure. No larger backbone, new frequency, Raw extraction, metadata predictor, hidden config, extra seed, patient exclusion or Outer evaluation. Clinical labels/counts enter the supervised FIT loss only. The additional margin is not calibrated uncertainty and patient-F1 training does not make repeated validation independent confirmation.
