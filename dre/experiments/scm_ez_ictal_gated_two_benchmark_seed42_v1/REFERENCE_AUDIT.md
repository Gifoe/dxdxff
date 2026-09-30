# Patient-reference audit

The patient reference is label-free. For each patient, synchronized record, state and frequency, it is the median across signal-valid channels. When at least three channels are valid, the current channel is excluded. With fewer than three valid channels, the all-valid median is used and counted as a fallback. EZ label, resection, outcome, center and patient identity never enter the reference operator.

`Self[j,k] = C[j] - C[k]` and `Patient[j,k] = C[j] - P_without_current[k]`. Invalid comparisons are zero and accompanied by two explicit 6x6 masks. Absolute spectra are not concatenated into the Full model.
