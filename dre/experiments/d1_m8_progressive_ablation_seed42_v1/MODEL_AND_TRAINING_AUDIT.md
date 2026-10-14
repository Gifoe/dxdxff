# Model and training audit

**VERIFIED:** A0/A1 share the original1,537-parameter CompactDelta topology, historical88-position duplicate/zero-aware LayerNorm moments, mapping multiplicities/structural zeros,96hidden GELU/dropout0.15 and final classifier. Native LayerNorm14 was not substituted. ABS14 occupies the same compact structural coordinates as DELTA14.

All fold seeds are42+1009*fold. A0/A1 initial parameter-state hashes match exactly; separately checked original D1 checkpoint binding initial hashes match in **all five folds**. Both new arms are scratch-trained, never loaded from trained D1. All ten cells actually completed129 saved epochs. No completed epoch was discarded, no candidate search or loss/calibration variant was added.

The unmodified pinned original trainer is called with source penalty0: AdamWlr0.001/wd0.0001, FIT-only NEZ-positive original BCE class weight, one channel-mean optimizer update per FIT patient, original patient permutation per epoch, clip1, max30/minstop6/patience6. Checkpoint key is validation patientMacroF1/EZF1/earlier epoch; original NEZ0.005 threshold grid/ties apply. Each scratch model chooses its own legal validation threshold. No SoftMacro objective is used for A0/A1.

A2 and A3 are original frozen models. A3 uses1,537 frozen D1 parameters plus9,481 correction parameters from its original training, whole-package residual cap1 and objective0.75weightedBCE+0.25patientSoftMacroF1. No weights, scales, preprocessing, architecture or thresholds of either reference are changed.

| Model | Fold | Selected epoch | Frozen NEZ threshold | New epochs run |
|---|---|---|---|---|
| A0 | 1 | 23 | 0.535 | 29 |
| A1 | 1 | 7 | 0.415 | 13 |
| A0 | 2 | 2 | 0.465 | 8 |
| A1 | 2 | 2 | 0.520 | 8 |
| A0 | 3 | 3 | 0.400 | 9 |
| A1 | 3 | 19 | 0.375 | 25 |
| A0 | 4 | 2 | 0.450 | 8 |
| A1 | 4 | 8 | 0.425 | 14 |
| A0 | 5 | 2 | 0.500 | 8 |
| A1 | 5 | 1 | 0.520 | 7 |
| A2 | 1 | 2 | 0.435 | reused |
| A2 | 2 | 19 | 0.260 | reused |
| A2 | 3 | 5 | 0.300 | reused |
| A2 | 4 | 3 | 0.320 | reused |
| A2 | 5 | 5 | 0.400 | reused |
| A3 | 1 | 4 | 0.410 | reused |
| A3 | 2 | 1 | 0.255 | reused |
| A3 | 3 | 10 | 0.295 | reused |
| A3 | 4 | 13 | 0.365 | reused |
| A3 | 5 | 15 | 0.395 | reused |

Independent verification confirms initial-state bindings, original selection metadata, BEST/LAST completion hashes, exact frozen reference outputs and all aggregate metrics. Three pretraining synthetic contract tests passed. All source byte hashes were locked before training.
