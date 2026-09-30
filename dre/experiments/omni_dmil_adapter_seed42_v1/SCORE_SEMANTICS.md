# Score semantics

- The frozen official CNN was trained with the opposite output orientation from the requested pathological-positive score.
- For every cached raw segment logit `l`, D-MIL uses `p_pathological = 1 - sigmoid(l)`.
- Channel baseline is `mean_j(p_pathological_j)`, algebraically identical to the audited official score flip after segment-wise sigmoid averaging.
- `sigmoid(mean(logit))` is prohibited and is never used.
- `pathological_labels == 1` is the positive channel class; `0` is normal and negative values are unlabeled.
- D-MIL reads no embeddings, waveforms, HFO features, electrode metadata, or patient-relative inputs.
