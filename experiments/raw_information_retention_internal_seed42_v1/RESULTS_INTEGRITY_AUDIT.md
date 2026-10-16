# Result-to-claim audit

Mode: numeric and claim audit. Artifacts: the21 anonymous server-exported
aggregates, executed-only report, figure sources and both rendered figures.

| Claim | Evidence | Assessment |
| --- | --- | --- |
| 53 cells and773 epochs actually completed | RUN_STATUS, cell table, independent terminal | Supported |
| Calibration score replay error0;908 patient result rows reconstructed | INDEPENDENT_VERIFICATION | Supported, engineering fidelity not predictive efficacy |
| First-path n=86 appearances/arm and59 unique patients | Independent verifier and matched cell counts | Supported; neither86 nor five paths are independent cohorts |
| Pre-latent reference improves ranking and EZ-F1 in the matched control | Paired component intervals | Supported within this model/protocol; no universal physiological necessity claim |
| Patient Z and temporal phases improve EZ-F1, not demonstrated ranking gains | Paired EZ-F1/AUROC intervals | Supported with the stated interpretation and no multiplicity-adjusted significance claim |
| Seizure SD is necessary | Its paired intervals span zero | Not supported; not asserted |
| C4 consistently beats the original baseline | C4-C0 EZ-F1 interval crosses zero | Not supported; report retains the negative conclusion |
| Raced improvements generalize across the additional internal paths | Raced-path interval crosses zero | Not established; overlapping paths are not independent replication |
| Longer receptive field alone caused a change | Width/depth/kernel/parameters also changed | Not identifiable; report calls this an encoder-family comparison |
| Relative amplitude is biologically useless | One normalization family did not improve this classifier | Not established; report explicitly separates preprocessing results from physiology |
| A patient-aware threshold can attain0.65 with these scores | Label-aware oracle is only a finite-sample upper bound and remains below0.65 on average | Not established; oracle is isolated from classifier results |
| EZ-F1>0.65 has been achieved | No new Outer; highest descriptive internal candidate0.404191 | False; target remains unachieved |

No literature citations or novelty claims were added. No missing classification
observations were generated. All nine arms are retained in the report/figures.
Source groups with fewer than five unique patients remain suppressed. The
original failed numerical comparison is retained as failed, separate from the
prospective deterministic execution.

Figure review: readable labels and all interval marks inspected. The ranking
legend was relocated after initial review because it overlapped the last row.
The final source preflight had13 passes, no failures and one reviewed parser
warning: its width scanner misread183/25.4 as183 inches. Actual SVG width is183mm
for both figures. `verify_release.py` independently checks that dimension.

No-invention status: PASS for executed aggregates and bounded claims.
No private rows/waveforms/checkpoints/identities/runtime logs are in this release.
