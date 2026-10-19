"""Bounded read-only installed-wheel and import diagnosis, no model fitting."""
import base64
import csv
import hashlib
import importlib.metadata as md
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(r'D:\drechan_temporal_pr_contrast_3seed_v1_runtime')
def main():
    tag=sys.argv[1] if len(sys.argv)>1 else 'legacy01'
    assert tag.replace('_','').isalnum()
    d=md.distribution('torch');checks=[]
    wanted=['torch/_C.cp313-win_amd64.pyd','torch/lib/torch_python.dll','torch/lib/torch_cpu.dll','torch/lib/torch_cuda.dll','torch/lib/c10.dll','torch/lib/c10_cuda.dll']
    records={r[0]:r for r in csv.reader(io.StringIO(d.read_text('RECORD')))}
    for n in wanted:
        p=Path(d.locate_file(n));r=records[n];h=hashlib.file_digest(p.open('rb'),'sha256').digest()
        actual=base64.urlsafe_b64encode(h).decode().rstrip('=');assert r[1]=='sha256='+actual and int(r[2])==p.stat().st_size
        checks.append(dict(file=n,sha256=h.hex(),bytes=p.stat().st_size,record_matches=True));print('WHEEL_FILE_PASS',n,flush=True)
    imports=[]
    for i in range(5):
        q=subprocess.run([sys.executable,'-X','faulthandler','-c',"import torch; print(torch.__version__); print(torch.arange(1024,dtype=torch.float32).sum().item())"],capture_output=True,timeout=60)
        for suffix,data in [('log',q.stdout),('err',q.stderr)]:
            target=ROOT/f'import_diagnostic_{tag}_{i+1:02d}.{suffix}';assert not target.exists()
            target.write_bytes(data)
        imports.append(dict(trial=i+1,exit_code=q.returncode,stdout_bytes=len(q.stdout),stderr_bytes=len(q.stderr)));print('IMPORT_TRIAL',i+1,q.returncode,flush=True)
    out=dict(status='DIAGNOSIS_COMPLETE',torch=d.version,python=sys.version.split()[0],files=checks,import_trials=imports,
        all_import_trials_pass=all(q['exit_code']==0 for q in imports),no_model_or_data_loaded=True,formal_updates=0)
    target=ROOT/f'public/RUNTIME_INTEGRITY_{tag}.json';assert not target.exists()
    target.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
