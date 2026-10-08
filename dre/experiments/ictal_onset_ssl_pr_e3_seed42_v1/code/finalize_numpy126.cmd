@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PYTHONFAULTHANDLER=1
set ROOT=E:\DRE-nips\new-pipeline\7-11\ictal_onset_ssl_pr_e3_seed42_v1
set RT=C:\ictal_onset_ssl_pr_e3_seed42_runtime
set PY=%RT%\metrics_numpy126_env\Scripts\python.exe
"%PY%" -c "import json,numpy,torch;from pathlib import Path;r=Path(r'%RT%');a=json.loads((r/'FINALIZATION_RUNTIME_AUDIT.json').read_text(encoding='utf-8-sig'));assert a['status']=='PASS' and a['metric_max_abs_diff']==0 and a['validation_logit_max_abs_diff']==0 and numpy.__version__==a['finalization_numpy'];assert all(torch.load(r/'validation_amended'/f'fold_{f}_{s}_last_private.pt',map_location='cpu',weights_only=False)['epoch']==e for f in range(1,6) for s,e in [('ssl',25),('E3',30)])"
if errorlevel 1 exit /b %errorlevel%
"%PY%" "%ROOT%\amended_code\run_e3_validation.py" --package "%ROOT%\amended_code" --export "%RT%\export_amended" --output "%RT%\validation_amended" --amendment "%ROOT%\PROTOCOL_AMENDMENT.json" --device cuda
exit /b %errorlevel%
