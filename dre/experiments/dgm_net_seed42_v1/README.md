# DGM-Net A2 — Seed42

Scope: latest user request **only A2**. Five original Development folds, frozen A0/D1 replay, frozen A2 interventions and10,000 paired unique-ID bootstrap draws. A1/A3/A4 are explicitly not trained; no full conditioning superiority claim or new Outer evaluation is authorized.

Architecture: original frozen D1 + independent-window CNN24 → same-channel pre-reference subtraction → bounded multiplicative seven-DELTA gate → early/late masked means → available-seizure mean/populationSD → PatientZ96 → residual head. No GRU/LSTM/Transformer/GNN/attention. Total11,019/trainable9,482 parameters.

`PROTOCOL_LOCK.json` fixes data hashes, original split, scale definition/floor, all five training schedules, interventions, effect-size thresholds and the limited scientific terminal before training. The original D1 source, FIT-only banks and Development-selected checkpoints are hash verified and reused, not modified. Local/private runtime locations are command-line configuration, not public data artifacts.

Trusted-server execution (NumPy imports before conda PyTorch; no unsafe OpenMP override):

```powershell
$env:DGM_BASE_CODE='E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code'
& 'E:\Anaconda\python.exe' -m pytest D:\dgm_net_seed42_v1\tests -q
& 'E:\Anaconda\python.exe' D:\dgm_net_seed42_v1\code\window_delta.py --runtime D:\dgm_net_seed42_v1_runtime --baseline D:\raw_delta_encoder_seed42_v1_exact_resume_runtime --source E:\DRE-nips\new-pipeline\7-11
& 'E:\Anaconda\python.exe' D:\dgm_net_seed42_v1\code\train.py --runtime D:\dgm_net_seed42_v1_runtime --baseline D:\raw_delta_encoder_seed42_v1_exact_resume_runtime --source E:\DRE-nips\new-pipeline\7-11
```

All new code/runtime reside on server D:. Original source/input locations remain read-only. Resume only the exact hash-bound command after inspecting process/checkpoints/logs; never duplicate running cells. Per-epoch model/optimizer/RNG/stopping state is atomic. Patient-level intervention results are private and checkpoint bound. No installed skills are used.

Completed: five A2 cells/95 epochs, frozen interventions and10,000 paired bootstrap draws. Read FINAL_REPORT for measured metrics, failed minimum-effect gates, unrun matched-control limitations and the universal conditioning-floor suppression issue. Scoped `.gitattributes` disables newline conversion to preserve published SHA256 bytes across platforms.

Publish source, hashes, tests and anonymous aggregates only. Raw, IDs, labels, banks, scalers, checkpoints, individual predictions and logs remain on the original trusted server. Results are repeatedly studied, selection-conditioned Development, not independent clinical confirmation. Read `results/RUN_STATUS.json` for actual execution state; unexecuted metrics are never fabricated.
