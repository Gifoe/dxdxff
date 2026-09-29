# Official Omni TimeConv-CNN reproduction gate

Status: **RAW_ENCODER_REPRODUCTION_FAILED** under the predeclared gate.
The graph/router work was not started.

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

## Frozen run and result

The server rebuilt 399 official-train EDFs into 65,060 60-s clips and 237
official-test EDFs into 240,074 60-s clips, with complete extraction/hash
audits. The pinned `cnn.py` SHA-256 was
`c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95`.
The ImageNet ResNet18 weights SHA-256 was
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Training completed all ten epochs; the sample-level internal validation F1
selected epoch 9 before the official test was scored. The frozen checkpoint
SHA-256 was
`442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852`.

| Official-test diagnostic | Published | This run | Frozen gate minimum |
|---|---:|---:|---:|
| Macro-F1, test-derived Youden | 0.6469 | **0.599267** | 0.6169 |
| Channel AUROC | 0.8061 | **0.798767** | 0.7761 |

The 8,104 labeled EDF-channel pairs consisted of 7,297 normal and 807
pathological pairs. Test-derived Youden threshold was 0.016663 for the
pathological score. At a fixed 0.5 threshold, macro-F1 was 0.659754, which
is a separately reported diagnostic. The predeclared criterion explicitly
required **both** Youden macro-F1 and AUROC. The fixed-threshold figure cannot
be substituted after seeing the test outcome. Thus the gate fails, despite
the near-published AUROC and the fixed-threshold diagnostic.

The pinned official evaluation source inverts both labels and probabilities,
computes the ROC/Youden threshold on the test set, then reports macro-F1 and
AUROC. Our evaluation follows that orientation and same unit of analysis;
there is no demonstrated label-orientation bug. The 300-Hz defaults in
`features.py` and `features_inference.py` conflict with the 1000-Hz CNN
configuration and paper. The official `train.py` CLI also defaults to five
epochs while the paper describes ten. These unresolved upstream provenance
discrepancies limit a claim of exact paper reproduction, but they do not
justify changing the already-locked 1000-Hz, ten-epoch run after seeing its
test result. No rerun or threshold tuning was performed.

This is an **exploratory repeated-test benchmark**, not a fresh blind
confirmation. The internal sample-level validation F1 was not treated as
independent generalization evidence. The exact gate criterion remains in
`GATE_CRITERION.json`, whose SHA-256 is
`259625f9121cbe846e2a3908204b594273f5706261aa5432231feb19d90b0f4e`.
