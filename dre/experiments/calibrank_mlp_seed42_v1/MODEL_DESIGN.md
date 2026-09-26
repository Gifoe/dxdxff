# Frozen-backbone CalibRank-MLP

All variants retain one exact historical B0 MLP and output one final EZ probability per channel. B0's 96-dimensional hidden state and EZ logit are precomputed from its frozen checkpoints and detached. The adapters contain no patient ID, center ID, test label or true EZ count. Fit-patient labels are used only in the loss; validation patients only choose an epoch and NEZ threshold.

- **B0:** unchanged historical patient-relative MLP and validation-selected threshold.
- **B1:** 13 label-blind patient-distribution statistics of detached B0 logits/evidence → LayerNorm → 16-wide MLP → predicted EZ fraction. A deterministic one-dimensional root correction shifts all channel logits equally so their average sigmoid matches that predicted fraction. Ordering is preserved.
- **B2:** detached 96-wide hidden state projected to 16 dimensions plus logit, percentile rank, entropy and normalized evidence count → 16-wide residual head. Zero-initialized final layer, `0.10*tanh` bounded residual, patient-mean centering, then deterministic uncertainty gate `4p(1-p)`. No global patient shift is available to this branch.
- **B3:** B2 ranking correction followed by B1-style predicted-prevalence calibration.

The original backbone never receives gradients from these heads. Adapter loss is `BCE + 0.10*SmoothL1(predicted-vs-true prevalence logit) + 0.05*patient hard-boundary ranking loss + 0.01*mean(residual²)`; absent branches contribute zero. AdamW uses LR `1e-3`, weight decay `1e-4`, max 30 epochs, patience 6 after epoch 6. Validation-patient Macro-F1 (then EZ-F1, then earliest epoch) selects the checkpoint and the original 0.005-grid threshold selector. The zero-initialized epoch-0 state reproduces B0 and is always eligible.

The private five-fold embedding cache is created by `code/prepare_b0_embeddings.py`; it is not a new model or an ensemble. The code's `--preflight-only` check verified B1/B2/B3 epoch-0 NEZ probabilities against B0 (maximum discrepancy `1.2e-7`) and finite nonzero fit-patient gradients before training.
