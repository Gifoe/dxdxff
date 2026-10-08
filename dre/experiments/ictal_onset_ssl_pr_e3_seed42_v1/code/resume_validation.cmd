@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PYTHONFAULTHANDLER=1
set PY=E:\Anaconda\envs\persist_stable_251\python.exe
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1
set RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
"%PY%" "%ROOT%\amended_code\run_e3_validation.py" --package "%ROOT%\amended_code" --export "%RT%\export_amended" --output "%RT%\validation_amended" --amendment "%ROOT%\PROTOCOL_AMENDMENT.json" --device cuda
exit /b %errorlevel%
