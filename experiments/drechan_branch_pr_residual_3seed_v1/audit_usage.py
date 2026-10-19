"""Post-fit unlabeled weight diagnostic; no waveforms, predictions or label I/O."""
import ctypes
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(0xffff0000))
import csv
import hashlib
import json
from pathlib import Path
import statistics
import torch

ROOT=Path(r'D:\drechan_branch_pr_residual_3seed_v1_runtime');HERE=Path(__file__).resolve().parent;PUB=ROOT/'public'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    frozen=jr(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')
    assert all(sha(HERE/n)==h for n,h in frozen['source'].items())
    collected=[];checked=[]
    for r in frozen['cells']:
        if r['arm'] not in ['FULL','CHANNEL_CONTEXT']:continue
        assert sha(Path(r['checkpoint']))==r['checkpoint_sha256'];q=torch.load(r['checkpoint'],weights_only=False,map_location='cpu')['model']
        final=q['head.2.weight'].detach().double().reshape(64)
        for branch in ['raw','feature']:
            coefficient=.25*q['gain_'+branch].detach().double().tanh();assert coefficient.shape==(64,)
            collected.append(dict(arm=r['arm'],seed=r['seed'],branch=branch,effective_gain_rms=float(coefficient.square().mean().sqrt()),
                                  maximum_abs_effective_gain=float(coefficient.abs().max()),final_weighted_gain_l1=float((final*coefficient).abs().sum())))
        checked.append(r['checkpoint_sha256'])
    assert len(checked)==30 and len(collected)==60;rows=[]
    for arm in ['FULL','CHANNEL_CONTEXT']:
        for branch in ['raw','feature']:
            for seed in ['ALL',42,43,44]:
                rr=[r for r in collected if r['arm']==arm and r['branch']==branch and (seed=='ALL' or r['seed']==seed)]
                rows.append(dict(arm=arm,branch=branch,seed=seed,models=len(rr),
                                 mean_effective_gain_rms=statistics.mean(r['effective_gain_rms'] for r in rr),
                                 minimum_effective_gain_rms=min(r['effective_gain_rms'] for r in rr),maximum_effective_gain_rms=max(r['effective_gain_rms'] for r in rr),
                                 maximum_abs_effective_gain=max(r['maximum_abs_effective_gain'] for r in rr),
                                 mean_final_weighted_gain_l1=statistics.mean(r['final_weighted_gain_l1'] for r in rr)))
    out=PUB/'RESIDUAL_USAGE.csv'
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit=dict(status='POSTFIT_UNLABELED_WEIGHT_DIAGNOSTIC',source_sha256=sha(HERE/'audit_usage.py'),freeze_sha256=sha(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json'),
               checkpoint_set_sha256=hashlib.sha256(json.dumps(sorted(checked)).encode()).hexdigest(),checkpoints=30,aggregate_rows=len(rows),output_sha256=sha(out),
               waveforms_or_labels_or_predictions_read=False,new_metrics_or_inference=False,weights_exported=False,original13_scientific_sources_unchanged=True,
               interpretation='Effective coefficient is0.25*tanh(g). Weighted L1 is an algebraic scale summary, not an observed activation/logit effect or a retrained ablation. No causal inference or independent selection evidence.')
    (PUB/'RESIDUAL_USAGE_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
