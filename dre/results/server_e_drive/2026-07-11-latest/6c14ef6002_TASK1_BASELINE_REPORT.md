# Task 1 Baseline Report

Task: channel-level EZ/NEZ localization on the frozen old-90 V3 patient folds.

Label direction: NEZ=1, EZ=0. Primary metric: patient_macro_f1.

Completed model/seed rows: 1. Recorded failures: 0.

| model | seed | accuracy | balanced_accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | precision_nez | recall_nez | f1_nez | precision_ez | recall_ez | f1_ez | AUROC_NEZ | AUPRC_NEZ | AUROC_EZ | AUPRC_EZ | MCC | TN | FP | FN | TP | n_channels | n_patients | patient_macro_f1 | patient_ez_f1 | patient_nez_f1 | patient_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_svm | 42 | 0.6720009079559641 | 0.5490063013726857 | 0.5493587073792361 | 0.5490063013726857 | 0.5491771671861158 | 0.6714133623380175 | 0.7832042882668255 | 0.7857783089333732 | 0.7844891871737509 | 0.31551312649164676 | 0.31223429381199813 | 0.3138651471984805 | 0.6078877734966374 | 0.8243641633114503 | 0.6078877734966374 | 0.3062766701959347 | 0.09836437747870005 | 661 | 1456 | 1434 | 5260 | 8811 | 90 | 0.49945027666212183 | 0.2581586220285806 | 0.7407419312956631 | 0.6668088277631903 |
