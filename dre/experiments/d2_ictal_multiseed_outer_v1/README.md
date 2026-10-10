# Ictal D2: fixed-split five-seed outer evaluation

Original D2 (`SourceA0`, rank4 source correction, 9221 parameters) with its
patient-relative `patient_z` preprocessing retained. This is not the no-PR
ablation and not an ensemble.

Predeclared seeds: **0, 1, 2, 3, 42**. Reuse the immutable original seed42
five-fold checkpoints and outer predictions; train the other four seeds from
scratch, with the exact original trainer. All seeds use identical fixed
FIT/validation/outer membership and the stored original FIT-only preprocessing.
Each seed/fold selects its own checkpoint and threshold on validation only.

Primary summary: compute the equal-patient outer mean over 80 patients / 7635
channels for each seed, then report the arithmetic mean and **sample standard
deviation (ddof=1)** across the five seed estimates. Do not average unequal-fold
means, select the best seed, or fuse predictions. Patient-cluster confidence
intervals resample the 80 patients' seed-averaged metrics; they are distinct from
seed SD. Five seeds do not constitute 400 independent patients.

This is an **exploratory repeated outer test**, on the same previously evaluated
cohort. It measures training-seed sensitivity under a fixed split, not new-cohort
or unseen-center generalization. No raw EEG extraction or test-driven changes.

## Run on the original Windows server

Use the same CUDA environment as the original D2:

```powershell
& 'C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe' `
  'E:\DRE-nips\new-pipeline\7-11\d2_ictal_multiseed_outer_v1\code\run_multiseed.py' all
& 'C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe' `
  'E:\DRE-nips\new-pipeline\7-11\d2_ictal_multiseed_outer_v1\code\verify_completed.py'
```

The runner hard-gates the original code/data/split/checkpoint hashes, verifies
bitwise FIT/validation preprocessing parity, locks all 25 selected models before
new outer inference, and seals every prediction before loading full-cohort
labels. Training can resume from original epoch checkpoints; completed scientific
evaluation cannot be run again. Final independent verification reads frozen
scores only and checks all metrics, means, SD and bootstrap intervals without
rerunning a model or changing any prediction.

Only source code and compact aggregate audits/results belong in this directory.
Patient/channel identities, predictions, checkpoints, caches, raw data and
runtime logs stay on the server.
