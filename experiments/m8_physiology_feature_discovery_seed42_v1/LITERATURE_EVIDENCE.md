# Literature-to-candidate map, checked 2026-10-16

Only primary research is used as methodological motivation. Different outcomes, montages, sampling rates, signals and validation populations prevent importing reported scores into this cohort. The list is a targeted evidence review, not an exhaustive systematic review.

| Primary source | What it supports | What is not licensed here |
|---|---|---|
| [O'Leary et al., Brain Communications 2026, fcag008](https://pmc.ncbi.nlm.nih.gov/articles/PMC13014369/) | Combines band-power trajectories over the first20s after seizure onset, supporting a prospective test of complementary frequency information and temporal summaries. | Small21-patient dataset and different ensemble design; no claim that its performance transfers. Its low-frequency and full high-gamma bands are not silently reconstructed from this cache. |
| [Bratu et al., Epilepsia2024, DOI10.1111/epi.17849](https://pubmed.ncbi.nlm.nih.gov/38041564/) | Peri-ictal permutation entropy, especially a postictal/ictal index, motivates testing a cheap ordinal-complexity measurement. | Our fixed peri-onset cache cannot reproduce its full postictal PEI. An early-ictal PE delta is a new candidate, not that published biomarker. It was less effective for some fast-onset patterns. |
| [Kozma et al., J Neurosci Methods2024,110180](https://pubmed.ncbi.nlm.nih.gov/38795977/) | Periodic/aperiodic spectral components can be complementary; aperiodic-only findings were insufficient for their surgical-outcome task. Motivates testing shape beyond absolute power with explicit negative expectations. | A simple restricted-bin slope is not full spectral decomposition, nor a certified excitatory/inhibitory balance estimate. Interictal surgical outcome is not identical to ictal channel localization. |
| [Weiss et al., Neurology2015, DOI10.1212/WNL.0000000000001656](https://pubmed.ncbi.nlm.nih.gov/25972493/) | Early high-gamma recruitment and temporal reference can relate to surgical localization; mechanism background only. | Full PLHG depends on low-frequency phase and wider high-gamma support unavailable under current trustworthy-data gates. It is excluded from the executable candidate list. |
| [Early-onset frequency/complexity study,2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3276710/) | Early-onset complexity and entropy responses differ by onset location/pattern; supports testing Hjorth/ordinal features as uncertain alternatives rather than universally useful biomarkers. | Does not establish superiority of Hjorth measures, and is not recent evidence by itself. No retrospective cherry-picking of seizure types or patients. |

## Current local evidence and limits

The authoritative local FCPA-LF data gate is `a0_fcpa_lf_seed42_v1/results/DATA_GATE.json`: failure of original cache provenance and adequate independently verified time/acquisition coverage. Nonzero delta/theta PSD does not bypass it. Only explicitly supported cache-derived shape measures are added; absolute biological interpretation remains conditional on the original acquisition limitations.

The exact historical feature source computes RMS as `sqrt(mean(x²)+1e-8)` and band values as `log1p(power)`. Those are facts verified in source, not yet a measured suppression fraction. With voltage-scale data, the additive constant may matter; current label-blind measurement will quantify it. Stable-log variance and amplitude-normalized shape candidates address numerical/unit sensitivity without altering the old measurements or rescoring Outer.

Finite extraction, correlation, coefficients, a linear probe and a favorable old Outer ablation are insufficient proof of feature necessity. Complete matched retraining, patient-disjoint feature selection, center stability and uncertainty remain mandatory.
