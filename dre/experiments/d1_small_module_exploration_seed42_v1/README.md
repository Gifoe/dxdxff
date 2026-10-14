# Frozen D1 small-module exploration — Seed42 Phase1

User-confirmed primary target: patient-average **Macro-F1 +absolute0.02**, from0.644546461935 to0.664546461935; inspect AUROC/AP/EZF1 and sensitivity. This is experimental exploration, not a promise that adding parameters guarantees two points.

Fixed before new training: three candidates×five original folds. M1 corrects the previous DGM conditioning floor without changing the topology; M2 adds722 parameters to preserve deterministic early/late7D DELTA statistics; M3 adds1354 parameters to learn nonlinear per-window DELTA co-occurrence before set/statistical pooling. All have the same frozen D1 and bounded residual identity, original training loss/schedule/selection, shared FIT-only magnitude scales, genuine masks/windows and patient-level evaluation.

Historical evidence: D2 and A2 added only~.006 Macro-F1; temporal CoTAR on older A0 did not reach a useful primary increment. PMA did not establish hidden-state magnitude benefit. Thus a large backbone, repeated attention/graph addition or merely copying a previously failed gate is not justified. These smaller candidates target specific all-window aggregation loss, and correct the measured unit-floor problem in a **new** prospective experiment. No original A2 artifact is changed.

General references: [Deep Sets](https://arxiv.org/abs/1703.06114) for permutation-invariant encoder/pooling, [FiLM](https://arxiv.org/abs/1709.07871) for conditioning. They do not prove clinical performance or novelty here.

Trusted-server command (all new files on D:):

```powershell
$env:DGM_COMPLETED_CODE='D:\dgm_net_seed42_v1\code'
$env:DGM_BASE_CODE='E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code'
& 'E:\Anaconda\python.exe' -m pytest D:\d1_small_module_exploration_seed42_v1\tests -q
& 'E:\Anaconda\python.exe' D:\d1_small_module_exploration_seed42_v1\code\explore.py --runtime D:\d1_small_module_exploration_seed42_v1_runtime
```

Only genuine precomputed Raw250-derived7D window physiology is reused; no EDF/Raw feature re-extraction, no new labels or metadata inputs, no baseline retraining and no new Outer TEST. Per-epoch model/optimizer/RNG/selection states are atomically checkpointed. Same-command resume requires identical input/code/source/scale/checkpoint hashes. Do not start duplicate jobs or modify unrelated GPU processes.

All candidates, failed gates, folds/centers, fixed-threshold interventions and10000 paired47-ID draws must be reported. Choosing a winner across validation-selected candidates is repeated, selection-conditioned Development, not independent confirmation. A failed phase does not authorize hidden rescue variants; further exploration needs its own prospective phase lock.

Only source/hashes/anonymous aggregate outputs may be exported. Weights, scalers, IDs, labels, raw/masks, per-channel predictions and logs remain on trusted server. Scoped attributes preserve actual source/protocol/output SHA256 bytes.
