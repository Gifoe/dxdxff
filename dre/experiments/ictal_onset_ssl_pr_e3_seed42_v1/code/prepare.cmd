@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PY=E:\Anaconda\envs\persist_stable_251\python.exe
set PKG=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1\package\ictal_onset_ssl_pr_v1
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1
set RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
"%PY%" -m pytest -q "%PKG%\tests" > "%RT%\package_tests.log" 2> "%RT%\package_tests.err"
if errorlevel 1 exit /b %errorlevel%
"%PY%" "%ROOT%\code\prepare_runtime_final.py" --package "%PKG%" --runtime "%RT%" > "%RT%\prepare_exhaustive.log" 2> "%RT%\prepare_exhaustive.err"
exit /b %errorlevel%
