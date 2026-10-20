# Frozen dual-view TRAIN/CAL diagnostic

Zero neural fits,zero deployment threshold changes,no ASSESS/Development/Outer scoring. These are diagnostic cell-equal summaries,not patient-average goal evidence. The selected checkpoints had already used all8CAL patients for epoch selection;leave-oneCAL threshold sensitivity is selection-conditioned,not independent validation.

| Body | Mean CAL Macro-F1 | Mean leave-oneCAL threshold Macro-F1 | Mean threshold SD | Mean changed held-CAL decisions |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.649321 | 0.631468 | 0.015733 | 0.015088 |
| PRUNED_REF | 0.634867 | 0.606413 | 0.025850 | 0.029004 |
| FULL | 0.624001 | 0.598332 | 0.033414 | 0.022197 |

## Initial-anchor standardized raw summaries

| Body/phase | Mean coordinate-mean RMS | Mean coordinate-SD RMS | Mean fraction abs(g)>5 | Mean fraction abs(g)>10 |
| --- | ---: | ---: | ---: | ---: |
| PRUNED_REF/TRAIN | 0.281459 | 1.089073 | 0.004065 | 0.000098 |
| PRUNED_REF/CAL | 0.399189 | 1.030589 | 0.004448 | 0.000197 |
| FULL/TRAIN | 0.672341 | 1.526341 | 0.021682 | 0.002570 |
| FULL/CAL | 0.763668 | 1.391356 | 0.020381 | 0.002391 |

Initial anchors were fitted with the untrained encoder,not the evolving selected encoder. Drift is descriptive and is not proof that normalization caused the F1 decline or that refitting anchors would improve it. No patient or coordinate is excluded by the5/10cutoffs.

Historical Raw-only margin-grid/patientZ/fixed-fraction calibration failed to provide reliable F1 gains;A0Gaussian posterior identification also failed. These are different bodies,not a theorem against all calibration,but prevent claiming a ready-made remedy. No new decoder or architecture is selected from this diagnostic.

Independent verification covers45selectedCALcells,360leave-oneCALthreshold/metric checks,60raw-statphase cells and75unlabeled-scorephase cells. Private derived arrays and identities stay on the server;only anonymous tables,hashes andthisreport are exported. Active goal remains unachieved.
