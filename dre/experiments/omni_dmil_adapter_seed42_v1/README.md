# Omni D-MIL Adapter seed42

This directory contains the preregistered, TRAIN-only D-MIL Adapter experiment for Omni-iEEG Task 2.

The terminal is `STOP_DMIL_TRAIN_GATE_FAILED`. No new official TEST evaluation was performed. Start with `FINAL_REPORT.md`, then inspect `TRAIN_GATE.json`, `TRAIN_OOF_SUMMARY.csv`, and `TRAIN_OOF_FOLD_METRICS.csv`.

`TRAIN_OOF_PREDICTIONS.csv` is deidentified: it contains an experiment-local anonymous unit index, center, label, frozen score-derived features, fold, model score, and frozen fold-train threshold. Patient IDs, EDF names, channel names, and dictionary-reversible tuple hashes are absent. Raw EEG, caches, checkpoints, runtime logs, and private prediction files are not included.
