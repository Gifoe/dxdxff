# Audited cache-to-PaReSet adapter

Input is the existing formal-cohort 28-D window-feature cache, **not** the raw EEG cache. The adapter is `build_patient_records.py`; its server-only output is `D:\nips-temp\pareset_ez_v1\patient_records_9d.pkl` (55,343,192 bytes). It writes neither raw EEG nor outcome fields.

1. Select the nine supplementary descriptors by exact feature name, with source indices `[0,1,3,4,5,7,8,9,10]`: delta/theta/beta/low-gamma/high-gamma log band powers, RMS, variance, line length per second and spectral entropy. Do not select by assumed consecutive order.
2. Map each seizure run's normalized channel names onto its patient's canonical `patient_index` order. Preserve explicit onset-relative `window_relative_centers_sec`; pad shorter runs/windows/channels with validity masks, rather than fabricated time points or feature values.
3. Use `patient_index.labels` exactly as the established server loader. Convert EZ-positive to NEZ-positive exactly once. Audit and disclose the 85 run-copy label disagreements; do not resolve them by outcome inspection.
4. The supplied `four_view_expansion` produces absolute, baseline delta, z-delta and ratio views (9×4=36). Its baseline uses only valid pre-onset windows. `fit_standardizer` is called on the fold's fit patients only.
5. Exclude surgery outcome, surgery success, center ID and patient ID from model input. IDs are used only to enforce the frozen membership and patient-equal reporting. No test metric is computed in preflight.

Checked invariants: 80/80 formal cohort members present, 256 selected runs, zero unmatched local channel names, zero nonfinite feature runs, zero nonmonotonic time arrays, pre- and post-onset windows on every selected run, label values restricted to `{0,1}`, and real-patient forward/backward finite. The original 28-D server feature construction is not numerically identical to the supplementary package's default Welch recipe; a comparison to old 28-D results is therefore descriptive only. Matched structural comparisons require re-running all controls on this same adapted cache.
