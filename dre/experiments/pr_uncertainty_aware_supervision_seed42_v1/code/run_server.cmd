@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
set PYTHON_EXE=C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe
set EXP_CODE=E:\DRE-nips\new-pipeline\7-11\pr_uncertainty_aware_supervision_seed42_v1\code
set EXP_RUNTIME=C:\pr_uncertainty_aware_supervision_seed42_runtime
"%PYTHON_EXE%" "%EXP_CODE%\runtime_admission.py" --runtime "%EXP_RUNTIME%" --source "E:\DRE-nips\new-pipeline\7-11"
if not "%errorlevel%"=="0" exit /b %errorlevel%
"%PYTHON_EXE%" "%EXP_CODE%\test_and_smoke.py" --runtime "%EXP_RUNTIME%" --source "E:\DRE-nips\new-pipeline\7-11" --tests "%EXP_CODE%\..\tests"
if not "%errorlevel%"=="0" exit /b %errorlevel%
"%PYTHON_EXE%" "%EXP_CODE%\run_development.py" --runtime "%EXP_RUNTIME%" --source "E:\DRE-nips\new-pipeline\7-11" --protocol "%EXP_CODE%\..\PROTOCOL_LOCK.json" --device cuda
exit /b %errorlevel%
