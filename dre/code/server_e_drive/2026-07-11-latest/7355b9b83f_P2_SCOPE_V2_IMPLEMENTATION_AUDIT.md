# P2-SCOPE-v2 Implementation Audit

P2-SCOPE-v2 retains the P2_TEMPORAL_Q10 feature backbone and original P2 loss. Its formal logit is `direct_nez_logit + scope_boundary_delta`; anchor, seizure, causal, center, and patient-shift residuals are not used in the SCOPE final prediction.

`direct_nez_logit` and `contextual_channel_embedding` come from the existing `channel_classifier` after B0, physics, temporal, and seizure aggregation. The boundary reranker receives only the contextual embedding, direct logit, detached direct rank, raw q10 probability, temporal/onset/spread deltas, and valid seizure count.

The fixed ledger builder reproduces the frozen five outer folds and creates one deterministic fit/validation split inside each outer training fold. The trainer rejects any test mismatch or overlap and never enters an inner-fold loop.

The cardinality prior is computed from fit labels only. The head receives only label-blind patient summaries. Outer-test decoding uses the Beta-Binomial predicted mode and top `K_hat` EZ scores; true K is not accepted by the formal decoder.
