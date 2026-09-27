# Implementation audit

- Exact source A1 tip `b2871b32873b67d0e6155d2766ef040ee1e9df01`, frozen source/input hashes and 150 validation grids checked. Other-12 VLOO selects the epoch; FIT and target use the same checkpoint.
- The source forward exposes `seizure_channel_embedding`; it is bitwise equal to the original CrossSeizureMILAggregator input. Frozen aggregator output and original classifier(R4) both replay source within 1e-6.
- Source dataset aligns local `channel_names_norm` to patient `canonical_channels` and carries `seizure_channel_mask`; both source hashes and runtime canonical names/masks were checked. Missing channels are excluded.
- All context construction, seizure shuffling, FIT PCA/scaler, FIT oracle fits, SVD and Ridge training occur without validation labels. Target directions/channel scores are frozen before target labels are read for the nondeployable oracle/evaluation.
- The SR_COMBINED shuffled control replaces only SR_DIST while retaining true persistence; a fake shuffled aligned-persistence control was intentionally not created. Thus SR_PERSIST cannot satisfy the ninth all-of-all identification condition.
- There are 65 fold-by-patient target cells but only 47 distinct validation patients. CIs resample patient-ID clusters, not cells; FIT association CI resamples FIT IDs within each selected-epoch coordinate. All pairwise/target data remain private.
- The legacy monolithic cache initializer materializes labels for all 80 patients before role filtering. Literal no-outer-label-materialization is false, although no outer loader, prediction, metric or outcome-based model choice was made. This remains exploratory development.
