# Eight raw iEEG baselines, with/without patient-relative latent normalization

Branch: `codex/m8-raw-backbones-pr-seed42-v1`.

This experiment trains **four raw backbones × two independent PR versions × all five frozen Outer folds**. These are not the preceding feature-only classifiers and not a D1/M8 logit fusion. No fifth backbone is evaluated. Source and protocol audits plus the CUDA admission tests are complete; real training is running on the trusted server. Final metrics must come only from completed frozen Outer passes, not this admission test.

## Sources and adaptations

- **TimeConv-CNN-adapted:** exact archived `OmniTimeConvCNNEncoder`, with STFT → 224 interpolated physical-frequency bins → TimeConv2D → full ResNet18 → 128D window embedding. Scratch initialization. This is not the official 60-s Morlet CNN reproduction. Interpolation does not create frequencies above the 125-Hz Nyquist.
- **EEGNet-adapted:** [authors' EEGNet implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py) guides a PyTorch EEGNet-8,2 translation. Temporal kernels 125 and31 at250Hz, depthwise/separable convolution, pooling4/8 and dropout0.5. A single input channel makes spatial convolution degenerate to a learned depthwise1×1 mapping; it never fixes or invents the patient's electrode count. Flattened temporal output is the window embedding. Spatial/head max-norm constraints retained.
- **SEEG-Net-adapted:** exact archived multiscale Conv1D, SE/residual block, bidirectional LSTM and temporal attention. Adapt `n_times=500`, `sfreq=250`; do not replace the recurrent encoder with a small CNN.
- **SEEGformer-adapted:** archived FFT frontend, real/imaginary/amplitude branches, CLS tokens and two masked cross-channel Transformer blocks per branch. No MRI or coordinates. Adapt500samples/fs250/FFT1000 and0.5–125Hz. Obtain per-channel embeddings before the original branch classifier topology; differentiable mean/SD pooling doubles each branch's head input64→128. Retain three heads and learned branch weighting. This is not the paper's complete original experimental protocol.

`configs/SOURCE_AUDIT.json` traces all vendored files to the repository's `dre/server_e_drive/SERVER_FILE_MANIFEST.csv`. The CRLF working copies match the manifest after LF normalization; runtime byte hashes are separately pinned. Archived runners were read but not reused blindly: their 200Hz/4s, window-level loss/aggregation and different inner splitting are not the current M8 contract.

## Data, aggregation and labels

The original patient-disjoint FIT/validation/Outer roles are reused, seed42. Each arm must cover80patients,256seizures,7,635canonicalpatient-channels (1,743EZ /5,892NEZ). Outer patients16/16/17/15/16; channels1631/1498/1746/1425/1335.

Use existing hash-verified Raw250 banks,500samples/window and **all1,471,965 genuinely valid channel-windows**. Original masks exclude padding/invalid boundaries and missing seizure channels. Centers are approximately−29to+29s. No new filtering, resampling, feature extraction, window sampling or patient exclusion.

All eight use the same channel-local per-window mean/population-SD standardization (floor1e-6, clip8). This is **not** cross-channel PR. The reference Raw cache's preprocessing is inherited, not a claim that it preserves new low-frequency bands.

Within each available seizure/channel, mean all genuine window embeddings. Across available seizures, compute mean and populationSD; interleave feature mean/SD to get one final channel embedding. This is the preceding Raw-ABS aggregation contract; no physiological DELTA or latent pre-reference subtraction. SEEGformer first uses its complete valid multichannel window for masked attention, then the same seizure/channel aggregation.

This ensures identical raw information and aggregation for **all eight new configurations**. It is not an architecture-only identical-input replacement of M8: M8 additionally uses a14D physiological DELTA base,7D guidance, pre-reference latent subtraction, early/late Raw pooling, residual architecture and macro-aware loss. M8/D1 are same-cohort/split reference comparisons, with those differences explicitly retained.

Internally the trusted original label coding is NEZ=1/EZ=0. Train NEZ logits with FIT-only class weight and report `y_EZ=1−y_NEZ`, `p_EZ=1−sigmoid(logit_NEZ)`. All positive-class results mean EZ. Each patient contributes exactly one channel-level score per canonical channel.

## PR placement and numerical convention

Original: Raw → backbone → full-window/seizure channel embedding → head.

PR: Raw → **same backbone and aggregation** → patient-wise feature-dimension populationZ → **same head**.

PR adds no parameters. Both versions independently train from identical fold-specific full initial state hashes. Use the entire patient's valid channel set, never a mini-channel batch. Float64 mean/populationSD; divide bySD ifSD>1e-8, otherwise1, exactly the M8 convention (not an additiveepsilon variant). Invalid channels are excluded. The operation uses no labels or cross-patient statistics, but does require unlabeled whole-patient channel context at inference.

## Training, selection and memory

See `configs/PROTOCOL_LOCK.json`: one fixed reasonable setting per backbone, no search or new seed. AdamW, original FIT patient-equal mean weightedBCE, max30/min6/patience6, gradient cap1. Backbones use lr0.0003/0.001/0.0005/0.0005 respectively, wd0.0001. Original/PR have identical optimizer, budget and fixed encoding chunk size.

Checkpoint/threshold selection uses only original legal validation patients. Thresholds maximize patient Macro-F1, then EZ-F1, then proximity0.5 and lower threshold on the shared0.005EZ grid; earlier epoch breaks checkpoint ties. This is the preceding feature-baseline selector with no forbidden auxiliary metric. It intentionally does not import the legacy M8 selector's auxiliary tie metric. No Outer-label fitting, calibration, model selection or threshold optimization is allowed.

CUDA BF16 trains the encoder/head; float32 reductions and float64 PR statistics. Validation/test are strictlyFP32, withTF32 disabled butcuDNN retained: an outcome-free synthetic probe rejected BF16 inference because encoding-chunk arithmetic drift was amplified by PR. Exact values and tests are in the admission audit. This change preceded all formal training; old preflight sources/logs remain private.

Full-patient **two-pass gradient caching** retains every genuine window: first encode deterministic physical chunks and record RNG; build complete channel embeddings/PR/head and differentiate; replay each chunk with its recorded RNG and embedding gradient. Restore the original first-forward BN buffers and final RNG so replay does not count twice. All encoder parameters train end-to-end. No frozen backbone, detached-learning shortcut, sampled patient, truncated gradient or chunk-local PR.

Both paired variants use fixed identical training chunks. BN/dropout make training chunking part of the locked training configuration; chunk-size-independent inference refers to evaluated full-patient PR/output, within verified FP32 numerical tolerance, not an assertion that all stochastic minibatch training trajectories are identical.

## Verified admission (actually run on RTX5090)

| Backbone | Parameters | Direct/cache gradient max error | FP32 encoding-chunk score max error |
|---|---:|---:|---:|
| TimeConv-CNN-adapted |11,336,737|1.1921e-7|9.4176e-6|
| EEGNet-adapted |2,329|1.5833e-8|2.2352e-7|
| SEEG-Net-adapted |279,950|2.9803e-8|1.1325e-6|
| SEEGformer-adapted |310,278|4.6567e-10|5.2155e-8|

Direct/cache logits and BN buffers matched exactly. Paired initial states match; padding is ignored; channel permutation equivariance and masked PR match the original populationZ convention. An entire legal FIT patient (64channels/11,328validchannel-windows) passed TimeConv backward in0.961s; this smoke state was discarded. **These are engineering admission measurements, not localization results.**

## Execution and artifacts

Trusted Windows uses the existing CUDA environment; all new data/checkpoints/logs are onD:. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; NumPy must load before Torch.

1. `python tests/smoke.py` gates all four before training.
2. `python run_all.py` runs TimeConv-CNN → EEGNet → SEEG-Net → SEEGformer, both versions and all five folds. No first-fold limit.
3. Each backbone freezes all ten selections before its single Outer pass; saves hash-bound per-patient score/embedding caches; seals all predictions before a new full Outer-label read.
4. Epoch-boundary model/optimizer/RNG/selection resume; completed cells/checkpoints/patient predictions are reused only with exact source/data/freeze bindings.
5. Each completed backbone produces actual anonymous main, extended, pooled, fold/center, foldmean±SD, ranking, metriccoverage, confusion, PRdelta and10,000seed42pairedpatient-bootstrap tables. Incomplete backbones do not get fabricated rows.
6. `tests/verify_results.py` independently verifies all eight configurations from actual frozen predictions, selection histories, metrics and paired bootstraps.

Patient/channel predictions, labels, individual metrics, latent embeddings, checkpoints, optimizer/scaler state, logs and identifying manifests **stay on the trusted server**. `public/` aggregate artifacts alone may be copied into repository`results/`. Anonymous aggregates keep center/fold summaries but no patient/channel IDs. Undefined metrics stayNaN with coverage counts. Pooled metrics, patient means and foldmean±SD remain separate.

Scientific status: **EXPLORATORY_REPEATED_TEST**. The cohort has already been inspected in prior research. All new configurations are locked before evaluation; this does not make the historical cohort a new blind confirmation. Do not change later backbones after viewing an earlier backbone's Outer result.
