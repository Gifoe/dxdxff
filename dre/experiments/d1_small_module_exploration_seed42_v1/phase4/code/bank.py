"""Extract frozen M8 FIT/VAL evidence only; no TEST loader."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'tools'))
from explore import context,make_base,Inputs,sha,read,save,write
from best_recipe import build

def features(final,base,latent,x):
    c=len(final);s=torch.sigmoid(torch.from_numpy(final)).numpy()
    ctx=np.r_[np.quantile(s,[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]),np.log1p(c),final.mean(),final.std(),base.mean(),base.std()]
    q=np.c_[final,base,(rankdata(s,method='average')-.5)/c,x,latent,np.repeat(ctx[None],c,axis=0)].astype(np.float32)
    assert q.shape==(c,129) and np.isfinite(q).all();return q

def input_binding(a):
    old=json.loads((a.previous/'INPUT_FREEZE_PRIVATE.json').read_text());assert old['code']=={p.name:sha(p) for p in (ROOT/'phase3/code').glob('*.py')}
    p1=json.loads((a.phase1/'INPUT_FREEZE_PRIVATE.json').read_text());assert p1['code']=={p.name:sha(p) for p in (ROOT/'code').glob('*.py')}
    upstream=Path(r'D:\dgm_net_seed42_v1\code');assert all(sha(upstream/f)==h for f,h in p1['DGM_code'].items())
    return dict(protocol=sha(ROOT/'phase4/PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},previous_inputs=sha(a.previous/'INPUT_FREEZE_PRIVATE.json'),previous_models=sha(a.previous/'MODEL_FREEZE_PRIVATE.json'),conditioning_registry=sha(a.data/'CONDITIONING_REGISTRY_PRIVATE.pkl'),phase1_inputs=sha(a.phase1/'INPUT_FREEZE_PRIVATE.json'))

def prepare(a):
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'results').mkdir(exist_ok=True);binding=input_binding(a);fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);fr=json.loads((a.previous/'MODEL_FREEZE_PRIVATE.json').read_text());store=Inputs(a.baseline,a.data,a.previous);store.arm='M1_SCALED_DGM';audits=[]
    for fold in range(1,6):
        ref=next(q for q in fr['selections'] if q['arm']=='M8_MACRO_AWARE' and q['fold']==fold);cp=a.previous/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt';bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';assert sha(cp)==ref['checkpoint_sha256'];scale_path=a.phase1/f'scale_fold{fold}_PRIVATE.pkl';scale_q=read(scale_path)
        d=torch.load(bp,map_location='cpu',weights_only=False);assert set(scale_q['fit_patients'])==set(d['patient'][d['train']]);assert set(d['patient']).isdisjoint(d['roles']['test']) if 'roles' in d else set(d['patient']).isdisjoint(banks[fold-1]['roles']['test'])
        bind=dict(input=binding,M8_checkpoint=sha(cp),D1_bank=sha(bp),scale=sha(scale_path));target=a.runtime/f'fold{fold}/BANK_PRIVATE.pkl'
        if target.exists():
            q=read(target);assert q['binding']==bind;audits.append(q['audit']);print('EVIDENCE_BANK_REUSED',fold,flush=True);continue
        base=make_base(core,fac,records,a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold);model=build(base).cuda().eval();ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
        scale=torch.tensor(scale_q['scale'],device='cuda');f=np.empty((len(d['x']),129),np.float32);ll=np.empty(len(d['x']),np.float32);bb=ll.copy();captured=[]
        hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy()))
        for n,pid in enumerate(sorted(set(d['patient'])),1):
            ii=np.flatnonzero(d['patient']==pid);raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');captured.clear()
            with torch.inference_mode():q=model(torch.as_tensor(d['x'][ii],device='cuda'),raw,mask,times,delta,scale)
            assert len(captured)==1;final=q[0].cpu().numpy();base=q[2].cpu().numpy();f[ii]=features(final,base,captured[0],d['x'][ii]);ll[ii]=final;bb[ii]=base
            store.cache.clear();store.raw.cache.clear();del raw,mask,times,delta,perms,q
            print('FIT_VAL_EVIDENCE',fold,n,'of',len(set(d['patient'])),flush=True)
        hook.remove();assert core.state_hash(model.state_dict())==core.state_hash(ck['model']);reference=read(a.previous/f'predictions_private/M8_MACRO_AWARE_fold{fold}.pkl')['modes']['full'];drift=float(np.abs(ll[d['val']]-reference['logit']).max());assert drift<=1e-6
        groups=[np.flatnonzero(d['patient']==p) for p in sorted(set(d['patient'][d['train']]))];mean=np.mean([f[ix].astype(float).mean(0) for ix in groups],axis=0);second=np.mean([(f[ix].astype(float)**2).mean(0) for ix in groups],axis=0);sd=np.sqrt(np.maximum(second-mean**2,0)).clip(1e-6);xx=((f-mean)/sd).astype(np.float32);assert np.isfinite(xx).all()
        audit=dict(fold=fold,validation_frozen_logit_max_difference=drift,FIT_patients=len(groups),validation_patients=len(set(d['patient'][d['val']])),features=129,patient_observable_only=True,scaler_fit_only=True,M8_checkpoint_sha256=sha(cp),D1_bank_sha256=sha(bp),scale_sha256=sha(scale_path))
        save(target,dict(binding=bind,audit=audit,x=xx,features_raw=f,base=ll,D1=bb,y=d['y'],patient=d['patient'],channel=d['channel'],center=d['center'],train=d['train'],val=d['val'],pre=dict(mean=mean,scale=sd,fit_patient_ids=sorted(set(d['patient'][d['train']]))),M8_threshold=ref['threshold']));audits.append(audit);del model,d,ck;torch.cuda.empty_cache()
    write(a.runtime/'results/EVIDENCE_BANK_AUDIT.json',dict(status='PASS',folds=audits,no_TEST_access=True,no_retraining_or_extraction=True));print('ALL5_FIT_VAL_EVIDENCE_BANKS_COMPLETE',flush=True)

def arguments(p):
    p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase4_runtime'));p.add_argument('--previous',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));return p
if __name__=='__main__':prepare(arguments(argparse.ArgumentParser()).parse_args())
