# Phase2 — Direct frozen-D1 error correction

Registered after fully completing Phase1, before any new phase2 training. No previous protocol, checkpoint or metric is replaced. Three new candidates×five original Development folds:257-parameter R14 correction,721-parameter stage-DELTA correction,9481-parameter morphology correction. All zero-initialize the last projection for exact initial D1 identity, use fixed bounded residual1.0*tanh(correction) and weighted BCE(final) only. This tests correction-oriented training rather than the previous auxiliary classification/zero scalar formulation.

```powershell
$env:D1_PHASE1_CODE='D:\d1_small_module_exploration_seed42_v1\code'
$env:DGM_COMPLETED_CODE='D:\dgm_net_seed42_v1\code'
$env:DGM_BASE_CODE='E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code'
& 'E:\Anaconda\python.exe' -m pytest D:\d1_small_module_exploration_seed42_v1\phase2\tests -q
& 'E:\Anaconda\python.exe' D:\d1_small_module_exploration_seed42_v1\phase2\code\run.py --runtime D:\d1_small_module_exploration_seed42_v1_phase2_runtime
```

All new runtime remains D:. The original D1 and Phase1 source/inputs/scales/outcomes are hash-bound. Every last/best epoch, optimizer/RNG/stopping state and private intervention replay is atomic. No duplicate process, new Raw extraction, additional seed, hidden configuration or new Outer TEST. Report all6 explored candidates globally and their exact failure/tradeoffs; never build a best-fold montage. Development is adaptive and repeatedly used, so no independent clinical confirmation claim.
