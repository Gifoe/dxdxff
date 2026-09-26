# B0 provenance audit

- Source: historical `task1_baselines/patient_controls.py`, model `patient_z_mlp`; original feature cache and frozen seed-42 checkpoints were reused, not retrained.
- Cohort: 80 patients, five frozen outer folds, 7,635 aligned channel records. B0 and archived CDEL ledgers match one-to-one on patient, fold, channel and label. The two historical split manifests are byte-identical (SHA-256 `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278`).
- Inputs: `p2_matched_simple`, 88 aggregated features (`results/B0_FEATURE_MANIFEST.json`). Labels are EZ=0, NEZ=1. This is the exact historical B0 feature set, but CDEL's constituent branches need not use the same feature set; equality of patients/folds/channels does **not** imply an identical-input architectural comparison.
- Preprocessing: patient-wise z-score, then fit-only imputer and standard scaler. Neither validation nor test fits these transforms.
- Backbone: LayerNorm(88) → Linear(88,96) → GELU → Dropout(0.15) → Linear(96,1); 8,817 trainable parameters.
- Historical optimization: AdamW, LR `1e-3`, weight decay `1e-4`, class-weighted BCEWithLogits (`pos_weight = #negative/#positive` in the training labels); seed `42 + 1009 × fold`; max 30 epochs, patience 6 after epoch 6. Checkpoint and a 0.005-grid threshold were selected using validation-patient Macro-F1 and its recorded tie-breaks.
- Five checkpoint replays on the server matched the archived per-channel B0 scores to maximum absolute error about `1.5e-7`, with exact channel and label alignment. The replay audit and all private records remain on the server.
- Historical B0 patient-equal seed-42 outcomes: Macro-F1 `0.616167`, EZ-F1 `0.423866`, NEZ-F1 `0.808469`, BA `0.664179`, EZ-AUPRC `0.531996`, EZ-AUROC `0.726844`. The historical by-seed file also contains a *pooled-channel* BA of `0.641686`; it is not the patient-equal BA quoted here.
- B0 selected epochs: folds 1–5 = `2, 1, 2, 1, 2`; validation-selected NEZ thresholds = `0.505, 0.225, 0.370, 0.420, 0.450`. Fold fit/validation/test patient counts = `51/13/16`, `51/13/16`, `50/13/17`, `52/13/15`, `51/13/16`.

Historical B0 prediction SHA-256: `3bd2039cc7d01ecdb375256354978066b4fc5b8adda1a58b84b2abb9f896`. Private patient/channel predictions and raw logits are not published. `results/DIAGNOSTICS_SUMMARY.json` records the remaining provenance hashes.
