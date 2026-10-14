"""Second prospectively locked exploration; all cells, immutable prior evidence."""
import argparse
import json
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parents[1]
PHASE1=Path(__import__('os').environ.get('D1_PHASE1_CODE',r'D:\d1_small_module_exploration_seed42_v1\code'))
sys.path.insert(0,str(PHASE1))
from explore import context,make_base,original_modules,Inputs,sha,read,save,write,load,EV
from direct_correction import FeatureCorrection,MorphCorrection
ARMS=['M4_FEATURE_CORRECTION','M5_STAGE_CORRECTION','M6_MORPH_CORRECTION']

def make(core,fac,records,cp,seed,arm):
    base=make_base(core,fac,records,cp,seed)
    return (MorphCorrection(base) if arm=='M6_MORPH_CORRECTION' else FeatureCorrection(base,arm)).cuda()

def admission(model,d,store,scale,core,arm):
    expected=dict(M4_FEATURE_CORRECTION=257,M5_STAGE_CORRECTION=721,M6_MORPH_CORRECTION=9481)[arm]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
    tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
    frozen=core.state_hash(model.base.state_dict());model.eval()
    with torch.inference_mode():q=model(x,raw,mask,times,delta,scale);assert torch.equal(q[0],q[2]) and torch.equal(q[1],torch.zeros_like(q[1]))
    pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'));opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
    grads=[]
    for step in range(3):
        model.train();opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=fn(q[0],y);loss.backward()
        assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ns={n:float(p.grad.norm().cpu()) for n,p in model.named_parameters() if p.grad is not None};assert ns['head.2.weight']>0
        if step==0:assert ns['head.0.weight']==0
        else:assert ns['head.0.weight']>0
        if arm=='M6_MORPH_CORRECTION':
            if step>=1:assert ns['cnn.0.weight']>0 and ns['gate.network.2.weight']>0
            if step>=2:assert ns['gate.network.0.weight']>0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();grads.append(dict(step=step+1,norms=ns))
    assert frozen==core.state_hash(model.base.state_dict()) and not model.base.training
    return dict(arm=arm,status='PASS',trainable=expected,total=1537+expected,initial_identity_exact=True,discarded_FIT_smoke_steps=3,base_unchanged=True,gradients=grads)

def fit(model,d,store,scale,cell,seed,binding,core,runtime,arm,fold):
    cell.mkdir(parents=True,exist_ok=True);bind=dict(input=binding,arm=arm,fold=fold,seed=seed,initial=core.state_hash(model.state_dict()),D1=core.state_hash(model.base.state_dict()),scale=scale.cpu().tolist())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);core.seed_all(seed);tr,va=d['train'],d['val'];pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'))
    best=key=None;stale=0;history=[];start=1;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind;model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng']);best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
        if done.exists():
            cc=json.loads(done.read_text());assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last);model.load_state_dict(best['model']);return best,history
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,31):
        t=time.perf_counter();model.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];pid=str(d['patient'][ii[0]]);raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=fn(q[0],y);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach().cpu()));del raw,mask,times,delta,perms,x,y,q,loss
        assert core.state_hash(model.base.state_dict())==bind['D1'] and not model.base.training
        pred=EV.predict(model,d,va,store,scale);s=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy();sel=core.select_threshold(d['y'][va],s,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(arm=arm,fold=fold,epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-t,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch));write(runtime/'results/RUN_STATUS.json',dict(status='PHASE2_TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],no_new_Outer=True));print('DIRECT_CORRECTION_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None;core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),threshold=best['threshold'],epoch=best['epoch']));model.load_state_dict(best['model']);return best,history

def main(a):
    pub=a.runtime/'results';pub.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());assert lock['models']==ARMS
    previous=json.loads((a.phase1/'results/RUN_STATUS.json').read_text());assert previous['status']=='ALL_PHASE1_EXPLORATION_COMPLETE'
    old=json.loads((a.phase1/'INPUT_FREEZE_PRIVATE.json').read_text());assert old['protocol']==lock['phase1_protocol'];assert old['code']=={p.name:sha(p) for p in PHASE1.glob('*.py')}
    binding=dict(protocol=sha(ROOT/'PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},phase1_input_freeze=sha(a.phase1/'INPUT_FREEZE_PRIVATE.json'),phase1_outcome=sha(a.phase1/'results/SCIENTIFIC_GATE.json'),source_code=old['code'],scales={str(f):sha(a.phase1/f'scale_fold{f}_PRIVATE.pkl') for f in range(1,6)})
    frozen=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if frozen.exists():assert json.loads(frozen.read_text())==binding
    else:write(frozen,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,b=original_modules();store=Inputs(a.baseline,a.data,a.runtime)
    rs=read(a.phase1/'D1_METRICS_PRIVATE.pkl');selects=[];histories=[];errors=[];audits=[]
    fr=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text())
    for arm in ARMS:
        store.arm='M1_SCALED_DGM' if arm=='M6_MORPH_CORRECTION' else 'M2_STAGE_DELTA'
        print('START_DIRECT_CORRECTION',arm,flush=True)
        for fold in range(1,6):
            seed=42+1009*fold;cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';r=next(q for q in fr['selections'] if q['arm']=='D1' and q['fold']==fold);assert sha(cp)==r['checkpoint_sha256'];bp=cp.parent.parent/'BANK_PRIVATE.pt';assert sha(bp)==r['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False)
            sq=read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl');assert set(sq['fit_patients'])==set(d['patient'][d['train']]);scale=torch.tensor(sq['scale'],device='cuda')
            if fold==1:
                model=make(core,fac,records,cp,seed,arm);audits.append(admission(model,d,store,scale,core,arm));write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS_SO_FAR',candidates=audits));del model
            model=make(core,fac,records,cp,seed,arm);initial=core.state_hash(model.state_dict());best,h=fit(model,d,store,scale,a.runtime/arm/f'fold{fold}/TRAIN',seed,binding,core,a.runtime,arm,fold)
            cpp=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';pb=dict(binding=binding,checkpoint_sha256=sha(cpp),threshold=best['threshold']);pp=a.runtime/'predictions_private'/f'{arm}_fold{fold}.pkl'
            if pp.exists():pred=read(pp);assert pred['binding']==pb
            else:pred=dict(binding=pb,modes={})
            modes=['full','residual0','shuffle','static']+[f'neutral_{k}' for k in range(7)]+(['identity'] if arm=='M6_MORPH_CORRECTION' else [])
            for mode in modes:
                if mode not in pred['modes']:pred['modes'][mode]=EV.predict(model,d,d['val'],store,scale,mode);save(pp,pred)
                score=torch.sigmoid(torch.from_numpy(pred['modes'][mode]['logit'])).numpy();rs.extend(m.patient_rows(c,d,score,best['threshold'],arm if mode=='full' else arm+'__'+mode,fold))
            assert np.array_equal(pred['modes']['full']['base_logit'],pred['modes']['residual0']['logit'])
            reference=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl');assert np.max(np.abs(torch.sigmoid(torch.from_numpy(pred['modes']['full']['base_logit'])).numpy()-reference['score']))<=1e-6
            for mode in modes[1:]:errors.append(dict(arm=arm,**EV.errors(d,pred['modes']['full'],pred['modes'][mode],best['threshold'],fold,mode)))
            selects.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],epochs=len(h),initial_state_sha256=initial,checkpoint_sha256=sha(cpp),D1_checkpoint_sha256=sha(cp),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            histories.extend(h);save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rs);pd.DataFrame(selects).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False);print('DIRECT_CORRECTION_FOLD_COMPLETE',arm,fold,flush=True);del model,d;gc.collect()
        store.raw.cache.clear();gc.collect()
    write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS',candidates=audits));write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selects))
    report=load('direct_correction_report',ROOT/'code/report.py');report.finish(a.runtime,a.phase1,c,rs,selects,errors,binding)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args()
    try:main(a)
    except Exception as e:
        write(a.runtime/'results/RUN_STATUS.json',dict(status='ENGINEERING_OR_INTEGRITY_FAILURE',exception_type=type(e).__name__,preserve_artifacts=True,no_new_Outer=True));raise
