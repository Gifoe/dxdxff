# Official Omni TimeConv-CNN reproduction gate

Status: **not yet measured**. Graph/router work must not start before this gate.

## Pinned primary sources

- [Official model and wavelet preprocessing](https://github.com/Omni-iEEG/Omni-iEEG/blob/57c22a75a59b5c3a98006806ad42000f6a3fa5b6/omni_ieeg/channel_model/channel_model_train/model/cnn.py)
- [Official model configuration](https://github.com/Omni-iEEG/Omni-iEEG/blob/57c22a75a59b5c3a98006806ad42000f6a3fa5b6/omni_ieeg/channel_model/channel_model_train/configs.py)
- [Official train](https://github.com/Omni-iEEG/Omni-iEEG/blob/57c22a75a59b5c3a98006806ad42000f6a3fa5b6/omni_ieeg/channel_model/channel_model_train/train.py)
- [Official channel-level evaluation](https://github.com/Omni-iEEG/Omni-iEEG/blob/57c22a75a59b5c3a98006806ad42000f6a3fa5b6/omni_ieeg/channel_model/benchmark/evaluation_channel.py)
- [Paper, Table 5 and Appendix E](https://arxiv.org/html/2602.16072)

The pinned CNN applies the official Morlet-like FFT wavelet transform at
10–300 Hz with 224 bins and image-wise min–max scaling to 0–255. Two temporal
Conv2d layers feed a modified ImageNet-pretrained ResNet18, followed by a 32-D
representation and binary head. The 224×224 interpolation is commented out
in the pinned source; it must not be silently added.

Paper and pinned `configs.py` use 1000 Hz, 60 s, 10 epochs, Adam 3e-4, batch
32, weighted sampling, BCE, and validation-F1 checkpoint selection. The pinned
`features.py` extraction functions instead default to 300 Hz even though
their input filter requires native frequency >900 Hz. This disagreement is an
upstream implementation discrepancy, not evidence that a 300-Hz run
reproduces the paper's 1000-Hz main benchmark. This gate uses 1000 Hz and
reports the discrepancy.

The published channel-level target is macro F1 **0.6469**, AUROC **0.8061**.
The official evaluation script obtains F1 at a test-derived Youden threshold;
this is reproduced only as a *diagnostic*, never used to choose model or
A1-NET threshold. The prior A1-v2 `macro_f1=0.593662` uses a frozen
train-side threshold and is not directly comparable to paper F1.

Source EDFs were removed after byte-verified native-signal cache migration.
The private cache retains native digital signals and EDF scaling metadata, so
waveforms can be reconstructed without using a feature checkpoint.
The original training feature extractor draws up to five random 60-s clips
without a fixed extraction seed. This reproduction uses a documented
deterministic seed42 per EDF so an interrupted extraction resumes exactly;
that is a provenance-controlled sampling deviation, not a label or split change.

Actual reproduction result, source rate checks, model/weight hashes, official
test F1/AUC, and gate decision will be filled after training/evaluation. If
the raw encoder is clearly below public benchmark after engineering checks,
the scientific terminal is `RAW_ENCODER_REPRODUCTION_FAILED` and no graph or
router training is authorized.
