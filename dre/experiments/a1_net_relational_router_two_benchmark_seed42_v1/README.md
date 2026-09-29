# A1-NET seed42: official raw-encoder gate

This branch reached its hard, independently auditable reproduction gate. The
official-style Omni TimeConv-CNN achieved channel AUROC 0.798767 but
test-Youden macro-F1 0.599267, below the frozen 0.6169 threshold. Its exact
terminal is `RAW_ENCODER_REPRODUCTION_FAILED`; graph/router training and
two-benchmark A1-NET evaluation were **not** run.

The original A1 and A1-TF results remain unchanged on their source branches.
All raw waveforms, converted segments, checkpoints, patient predictions and
runtime logs stay on the private Windows server, outside Git.

See `PROTOCOL_LOCK.json`, `GATE_CRITERION.json`,
`OMNI_CNN_REPRODUCTION.md`, and `FINAL_REPORT.md` for the source-level audit,
frozen criterion, results and stop decision.
