@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PY=E:\Anaconda\envs\persist_stable_251\python.exe
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1
set LIB=%ROOT%\amended_code
set RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
"%PY%" -m pytest -q "%LIB%\tests" > "%RT%\amended_package_tests.log" 2> "%RT%\amended_package_tests.err"
if errorlevel 1 exit /b %errorlevel%
"%PY%" "%LIB%\test_resume_parity.py" --package "%LIB%" --output "%RT%\RESUME_PARITY_AUDIT.json" > "%RT%\resume_parity.log" 2> "%RT%\resume_parity.err"
if errorlevel 1 exit /b %errorlevel%
if exist "%RT%\export_amended\audit.json" goto train
"%PY%" "%LIB%\audit_export.py" --feature-cache "D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl" --raw-cache "D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl" --frozen-folds "%RT%\frozen_folds.json" --expected-sha-file "%ROOT%\package\ictal_onset_ssl_pr_v1\expected_sha.json" --output "%RT%\export_amended" --allow-source-only --approved-exclusion-private "%RT%\PADDING_RECORDS_PRIVATE.json" --amendment-json "%ROOT%\PROTOCOL_AMENDMENT.json" > "%RT%\export_amended.log" 2> "%RT%\export_amended.err"
if errorlevel 1 exit /b %errorlevel%
:train
"%PY%" "%LIB%\run_e3_validation.py" --package "%LIB%" --export "%RT%\export_amended" --output "%RT%\validation_amended" --amendment "%ROOT%\PROTOCOL_AMENDMENT.json" --device cuda
exit /b %errorlevel%
