# DRST-PR S4-only feasibility (exploratory)

Exact historical A1 replay: AP 0.576743 (pass). No A1 retraining.

S4 fixed lr=3e-4, wd=1e-3: AP 0.533233; Δ vs A1 -0.043510 (95% patient-ID bootstrap [-0.095622, +0.008429]); positive folds 1/5.

Decision: `STOP_NO_FUSION`. Error complementarity: A1-wrong/S4-right top1 4, reverse 14.

No S1/S2/S3 attribution, A1 replicate, ensemble control, or unique-complementarity claim. A1 used historical VLOO selection while S4 used FIT-only selection. Historical A1 target outcomes had already been viewed; this is not sealed confirmation. No outer test was accessed.
