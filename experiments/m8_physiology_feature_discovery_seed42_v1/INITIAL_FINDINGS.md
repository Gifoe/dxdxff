# Initial feature-discovery evidence — not final M8 results

Date:2026-10-16. The requested full feature-discovery goal remains active. This turn made verified progress (new code,actual extraction,615 completed FIT-only linear screens and independent artifact checks), not merely a status restatement. No new M8 candidate model or Outer classification was run in this phase.

## Actual completed work

| Evidence | Actual result | What it does not establish |
|---|---|---|
| Original7 window DELTA versus frozen R14 | Maximum reaggregation error0 | Does not validate a new classifier or certify clinical acquisition filters |
| New candidates | 11 window measurements,22 seizure-summary coordinates | Not11 proven SOZ biomarkers |
| Coverage |80patients,7635channels,256seizures,1471965valid channel-windows; finite100% | Correlated windows are not independent sample size |
| Original7 linear probes |225:15variants×3patient-disjoint screens×5originalFIT cohorts | Not full-M8 feature-necessity proof |
| Candidate linear probes |390:26variants×3screens×5FIT cohorts | No original validation/Outer feature selection; no final feature winner |
| Independent score replay |390 cached model/scaler pipelines reproduced probabilities with maximum error0; threshold/role/sklearn checks PASS | No newly fitted neural model verified |
| Independent candidate aggregation | All80 cached patients reaggregated with original aggregation code; max error0 | Remaining nonlinear/center/generalization experiments still required |

## Numerical issues actually measured

The exact old source uses `RMS=sqrt(mean(x²)+1e-8)`. Across all valid source windows:

- Median fraction of the squared RMS expression contributed by the added constant: **0.7047296**.
- **70.8583%** of windows have constant contribution≥50%; **16.8045%** have contribution≥90%.
- Median across80patient-specific medians: **0.6916064**.
- Independent direct-waveform formula spot checks on six preselected patients had max error0.

This quantifies a substantial numerical/unit-dependent component. It does **not** establish that the resulting RMS delta is constant or predictively useless. No baseline formula was silently changed.

In original fold1's FIT cohort (51patients), the existing Patient-Z SD≤1e-8 fallback is triggered for beta mean/std in86.27%/94.12%, low-gamma in96.08%/94.12%, and partial high-gamma in98.04%/98.04% of patients. All five FIT cohorts are retained in `OLD7_NUMERIC_FIT_AUDIT.csv`; this paragraph is explicitly fold1, not a pooled estimate. FIT-only StandardScaler and downstream nonlinear coupling may still recover useful information; these rates alone do not justify dropping bands.

RMS/variance/energy redundancy and scale sensitivity need a numerical-control experiment. A useful **next prospective control** is replacing only legacy `log1p(V²)` band outputs by stable log-power before the same pre-reference subtraction, while keeping frequency support unchanged. It must be separately locked and compared without rewriting original7 features or using Outer outcomes. This is a preprocessing/control hypothesis, not a new biological biomarker claim.

## Provisional candidate signals — original FIT only

Under the predeclared linear eligibility gate, only the following fold-specific additions passed. Each fold makes its own decision; these rows must not be merged into one globally selected feature set.

| Original outer-fold FIT | Candidate | Supporting screens /3 | ΔAUROC | ΔAP | ΔMacro-F1 |
|---|---|---:|---:|---:|---:|
|1|Clipped excess kurtosis|2|+0.006722|+0.009580|+0.000056|
|2|Spectral centroid13–95|3|+0.008037|+0.008410|+0.024156|
|2|Partial high-gamma fraction80–125|2|+0.009191|+0.009056|+0.020026|
|4|Clipped excess kurtosis|2|+0.005652|+0.007688|+0.003002|
|5|Spectral centroid13–95|2|+0.007691|+0.002421|+0.015082|

No candidate passed in fold3; no candidate passed all five FIT cohorts. This is an immediate warning against claiming a universal new marker from this screen. The full candidate list,including unfavorable results,is preserved. Each addition's comparator is old7 under the same linear probe,not the trained M8 network.

The original7 deletion probes also suggest conditional redundancy for some amplitude/spectral coordinates,while deleting line length tends to hurt the linear discriminator. These are hypotheses for matched nonlinear retraining; do not promote them to biological conclusions or choose a cross-fold global deletion.

## Source/data restrictions

The historical low-frequency fidelity gate remains failed. No delta/theta/full-alpha recovery is claimed or attempted. No full80–150Hz HFO/PLHG or low-frequency phase coupling is reconstructed from250Hz Raw. The slope candidate is explicitly a descriptive restricted-bin proxy,not a validated aperiodic/E-I estimate. Permutation entropy here is not the published full postictal PEI.

Only code,hashes and anonymous summaries were transferred. Existing raw data,IDs,feature arrays,classifiers/scalers,individual scores and logs remain private. New model integration,center-held-out analyses,nonlinear ablations and final performance verification are **pending**; see `WORKPLAN.md` for the complete unreduced objective and completion audit.
