# Frozen compact feature contract

The feature archive/order/split/manifest hashes are in `PROTOCOL_LOCK.json`.
All88 names must match the original primary/physics view grammar. ABS18,
DELTA26, ZDELTA26 and RATIO18 are reference views; **DELTA is not the1–4Hz band**.
The old export has16 exact auxiliary duplicates and16 constant-zero spectral
coordinates. Copies of a removed view are removed together; no deduplication,
new PSD extraction, frequency addition or change to the88D interface occurs.
Nominal retained columns are not independent varying physiological dimensions.

| Arm | Retained | Columns retained | ZDELTA |
|---|---|---:|---|
| F0/F1/F5/F7 |ABS+DELTA+RATIO|62|Absent26 |
| F2 |ABS|18|Absent26 |
| F3 |ABS+RATIO|36|Absent26, DELTA also absent26 |
| F4 |ABS+DELTA|44|Absent26, RATIO also absent18 |
| F6 |all four|88|Restored26 |

F0 and F2–F5 keep original Patient Z and FIT-only imputer/scaler before masking.
F1 removes only patient_z, uses the original separately FIT-fitted absolute
imputer/scaler, then applies the same62-column compact mask. F1 retained
values therefore match its legal noZ transform, not the Patient-Z F0 values.
Other retained values must match original F0 exactly on FIT/VAL and inference.
F5 replaces only native input LayerNorm with Identity (8641 parameters),
keeping all shared original linear initialization. Every new model is scratch
trained, not fine-tuned from a fitted F0. Zero masking changes input-dependent
LayerNorm statistics as well as feature information; the retraining effects
are not causal physiological feature importance.
