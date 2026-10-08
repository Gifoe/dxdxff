@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PYTHONFAULTHANDLER=1
set PY=C:\ictal_onset_ssl_pr_e3_seed42_runtime\metrics_numpy126_env\Scripts\python.exe
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e1_seed42_v1
set LIB=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1\amended_code
set E3RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
set RT=C:\ictal_onset_ssl_pr_e1_seed42_runtime
"%PY%" "%ROOT%\code\test_e1_parity.py" --package "%LIB%" --output "%RT%\E1_PARITY_AUDIT.json" > "%RT%\parity.log" 2> "%RT%\parity.err"
if errorlevel 1 exit /b %errorlevel%
"%PY%" "%ROOT%\code\run_e1_validation.py" --package "%LIB%" --export "%E3RT%\export_amended" --output "%RT%\validation" --protocol "%ROOT%\PROTOCOL_LOCK.json" --e3-output "%E3RT%\validation_amended" --device cuda
exit /b %errorlevel%
