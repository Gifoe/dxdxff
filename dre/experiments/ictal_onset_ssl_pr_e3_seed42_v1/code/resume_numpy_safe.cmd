@echo off
rem REJECTED diagnostic: disabling AVX2/FMA3 did not fix the synthetic crash.
rem Do not use this route for finalization; see finalize_numpy126.cmd and audits.
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PYTHONFAULTHANDLER=1
set NPY_DISABLE_CPU_FEATURES=AVX2,FMA3
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1
set RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
set PY=E:\Anaconda\envs\persist_stable_251\python.exe
"%PY%" "%ROOT%\amended_code\audit_numpy_dispatch.py" --package "%ROOT%\amended_code" --output "%RT%\numpy_dispatch_safe_private.json" --compare "%RT%\numpy_dispatch_baseline_private.json"
if errorlevel 1 exit /b %errorlevel%
call "%ROOT%\amended_code\resume_validation.cmd"
exit /b %errorlevel%
