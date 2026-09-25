# Cache Key Mapping

This audit describes storage only and never defines either task cohort.

- Feature records: 501; patients: 155
- Raw records: 501; patients: 155
- Feature tensor: `sample.window_features` with `[window, channel, feature]`.
- Raw tensor: `sample.raw_waveform` with `[channel, time]`; window centers are `sample.window_relative_centers_sec`.
- Channel names: `record.channel_names_norm`.
- Task 1 authoritative channel labels: `patient_index[subject].canonical_channels + labels`; record-aligned labels are fallback only when canonical labels are absent.
- Task 2 outcomes: resolved only through `outcome_hifos.outcome_resolver`.
