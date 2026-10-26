# Patient reference transfer: frozen protocol

The machine-readable `PROTOCOL_LOCK.json` defines every numerical choice. This is
a new exploratory experiment on all80 historical patients, not confirmation in
new patients. It does not replace the earlier unsuccessful transferable-SNR probe.

## Question and design

Can unlabeled channels estimate a patient's location/scale that predicts other
channels, beyond a global or same-source reference? Does relative atypicality still
associate with EZ? Statistical background is not assumed healthy or irrelevant.

Use the original12 active R14 coordinates and the exact audited80-person parent
assignment (79 original FIT,1 original-validation exception). No classifier or raw
extraction. Common legal TRAIN preprocessing is frozen.100 label-free hash-based
channel halves are exchanged; each channel is a query once per repetition.

The five Gaussian marginal probes are GLOBAL, SOURCE, CENTER, SCALE and FULL.
CRPS is computed on identical held-out measurements in identical units. It is not
computed separately on each arm's rescaled output. Zero SD is a point forecast,
whose loss is absolute error; constants are retained. Gaussian probes need not be
true distributions to compare their held-out predictive errors.

Primary gain is GLOBAL minus FULL CRPS. Six secondary slots decompose centering,
scaling,source adjustment and the Full EZ-minus-NEZ mean absolute relative gap.
The anomaly support threshold is the operational TRAIN-unit1e-8 rule (unsupported
coordinate output0), reported explicitly and separate from CRPS's exact SD0 rule.
Full-channel operational parity includes the original final TRAIN scaler; the
cross-channel probe itself has no separate arm scaler.

## Inference and limitations

First average100 repeats and all channels within each patient. All80 patients
then count equally.10000 seed42 paired,within-source patient bootstrap draws give
95% primary/nominal intervals and additional six-slot Bonferroni intervals.
All sources and coordinates remain descriptive; no selecting winning subsets.
1000 within-patient label shuffles yield a conditional reference distribution,
not an independent population permutation test. Shared TRAIN uncertainty is not
estimated by this conditional bootstrap. Electrode anatomy and sampling may
contribute to a patient reference; this cannot establish biological causality.

The strong interpretation requires supported primary gain, SOURCE-FULL gain and
positive EZ atypicality. Distribution gain alone is insufficient. A nonsignificant
FULL-CENTER difference is unresolved, not proof of equal effects. Seven single-
seizure patients and their constant SD coordinates remain in all primary analyses.

## Lock and release

Admission, reference freezing, endpoint sealing, independent verification and
aggregate reporting are separate barriers. Code/protocol are pushed before real
outcomes. Frozen scientific files remain byte-identical. Output is anonymous
aggregates only, with one four-panel vector figure and600dpi PNG.

CRPS reference: Gneiting and Raftery(2007),
https://doi.org/10.1198/016214506000001437.
