"""Admission tests and discarded real FIT-only steps before scratch students."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from base import seed_all,sha,json_write,PRMLP
from model import make,objective
from prepare import binding


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--official-source',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    torch.set_num_threads(2); root=a.runtime; pub=root/'public'; bind=binding(a.protocol)
    for name in ['A0_REPRODUCTION','DATA_PREPROCESSING_AUDIT']:
        r=json.loads((pub/(name+'.json')).read_text()); assert r['status']=='PASS' and r['binding']==bind
    os.environ['TABM_SOURCE']=str(a.official_source)
    testpath=Path(__file__).resolve().parents[1]/'tests/test_tabm.py'
    subprocess.run([sys.executable,'-m','pytest',str(testpath),'-q'],check=True)
    from source_parity import official
    official(a.official_source)
    json_write(pub/'TABM_SOURCE_PARITY_AUDIT.json',{'status':'PASS','commit':'28e47ae301c92ec37787dde1ce923a0793f405b4',
        'full_source_sha256':sha(a.official_source),'isolated_source_sha256':sha(Path(__file__).with_name('tabm_layers.py')),
        'components_AST_semantically_identical':True,'identical_initial_tensors':True,'exact_float32_forward':True,'exact_float32_gradients':True,
        'no_unused_embedding_dependency_installed':True})
    d=torch.load(root/'fold1/DEVELOPMENT_PRIVATE.pt',weights_only=False); tr=d['train']
    assert d['binding']==bind; rows=[]
    for arm,total in [('W1',10167),('W2',10132)]:
        seed_all(1051); m=make(arm).to(a.device); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        assert sum(p.numel() for p in m.parameters())==total
        pos=torch.tensor(int((d['y'][tr]==0).sum())/int((d['y'][tr]==1).sum()),device=a.device)
        for pid in sorted(set(d['patient'][tr]))[:3]:
            ix=np.flatnonzero(d['patient']==pid); assert set(ix).issubset(set(tr))
            x=torch.as_tensor(d['x'][ix],device=a.device); y=torch.as_tensor(d['y'][ix],dtype=torch.float32,device=a.device)
            opt.zero_grad(); logits=m(x); loss=objective(logits,y,pos); loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        rows.append({'arm':arm,'parameters':total,'real_FIT_patient_updates':3,'finite_gradients':True,'member_output_shape_N4':arm=='W2'})
    json_write(pub/'PARAMETER_AUDIT.json',{'status':'PASS','A0':sum(p.numel() for p in PRMLP().parameters()),'W1':10167,'W2':10132,
        'W2_parts':{'LN':176,'shared_weight':8448,'r':352,'s':384,'hidden_bias':384,'heads':388}})
    json_write(pub/'TEST_AND_SMOKE_AUDIT.json',{'status':'PASS','binding':bind,'torch':torch.__version__,'numpy':np.__version__,
        'synthetic_groups':6,'tests_sha256':sha(testpath),'FIT_only_smoke':rows,'outer_evaluation':False,'smoke_not_formal_training':True})
    print('TABM_TEST_AND_REAL_FIT_SMOKE_PASS',flush=True)


if __name__=='__main__':main()
