@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
rem Fresh reconstruction template only; do not overwrite the completed private gate.
"C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe" -u -X faulthandler "E:\DRE-nips\new-pipeline\7-11\pr_uncertainty_aware_supervision_seed42_v1\code\reproduce_b0.py" --source "E:\DRE-nips\new-pipeline\7-11" --cache "D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl" --ledger "D:\nips-temp\task1_patient_relative_controls_v2\oof_ledgers\patient_z_mlp\seed_42_channel_oof.csv" --split "D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv" --manifest "D:\nips-temp\task1_patient_relative_controls_v2\configs\feature_manifest.json" --checkpoints "D:\nips-temp\task1_patient_relative_controls_v2\checkpoints\patient_z_mlp\seed_42" --output "C:\pr_uncertainty_aware_supervision_seed42_runtime\gate_fresh_reconstruction"
exit /b %errorlevel%
