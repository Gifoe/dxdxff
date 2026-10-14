"""Fixed decision-targeted Phase3; preserve all prior frozen trials."""
import argparse
import json
import sys
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[1]
P2=Path(__import__('os').environ.get('D1_PHASE2_CODE',r'D:\d1_small_module_exploration_seed42_v1\phase2\code'))
sys.path.insert(0,str(P2))
from run import context,make_base,original_modules,Inputs,sha,read,save,write,load,EV
from direct_correction import MorphCorrection
from decision_modules import MarginGated,objective
ARMS=['M7_MARGIN_GATED','M8_MACRO_AWARE']

def make(core,fac,records,cp,seed,arm):
    base=make_base(core,fac,records,cp,seed)
    return (MarginGated(base) if arm=='M7_MARGIN_GATED' else MorphCorrection(base)).cuda()

def admission(model,d,store,scale,core,arm):
    expected=9483 if arm=='M7_MARGIN_GATED' else 9481
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
    tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid];raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
    frozen=core.state_hash(model.base.state_dict());model.eval()
    with torch.inference_mode():q=model(x,raw,mask,times,delta,scale);assert torch.equal(q[0],q[2])
    pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'));opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);grads=[]
    for step in range(3):
        model.train();opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=objective(q[0],y,fn,arm);assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ns={n:float(p.grad.norm().cpu()) for n,p in model.named_parameters() if p.grad is not None};assert ns['head.2.weight']>0
        if step>=1:
            assert ns['head.0.weight']>0 and ns['cnn.0.weight']>0 and ns['gate.network.2.weight']>0
            if arm=='M7_MARGIN_GATED':assert ns['margin_a']>0 and ns['margin_b']>0
        if step>=2:assert ns['gate.network.0.weight']>0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();grads.append(dict(step=step+1,norms=ns))
    assert frozen==core.state_hash(model.base.state_dict())
    return dict(arm=arm,status='PASS',trainable=expected,total=1537+expected,initial_identity_exact=True,discarded_legal_FIT_smoke_steps=3,base_unchanged=True,gradients=grads)

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
            opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=objective(q[0],y,fn,arm);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach().cpu()));del raw,mask,times,delta,perms,x,y,q,loss
        assert core.state_hash(model.base.state_dict())==bind['D1'] and not model.base.training
        pred=EV.predict(model,d,va,store,scale);s=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy();sel=core.select_threshold(d['y'][va],s,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(arm=arm,fold=fold,epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-t,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch));write(runtime/'results/RUN_STATUS.json',dict(status='PHASE3_TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],no_new_Outer=True));print('DECISION_MODULE_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None;core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),threshold=best['threshold'],epoch=best['epoch']));model.load_state_dict(best['model']);return best,history

def main(a):
    pub=a.runtime/'results';pub.mkdir(parents=True,exist_ok=True);assert json_load(ROOT/'PROTOCOL_LOCK.json')['models']==ARMS
    old=json_load(a.phase2/'INPUT_FREEZE_PRIVATE.json');assert old['code']=={p.name:sha(p) for p in P2.glob('*.py')};assert json_load(a.phase2/'results/RUN_STATUS.json')['status']=='ALL_PHASE2_DIRECT_CORRECTION_COMPLETE'
    binding=dict(protocol=sha(ROOT/'PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},phase2_input=sha(a.phase2/'INPUT_FREEZE_PRIVATE.json'),phase2_outcome=sha(a.phase2/'results/SCIENTIFIC_GATE.json'),prior_code=old['code'],phase1_scales={str(f):sha(a.phase1/f'scale_fold{f}_PRIVATE.pkl') for f in range(1,6)})
    frozen=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if frozen.exists():assert json_load(frozen)==binding
    else:write(frozen,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,b=original_modules();store=Inputs(a.baseline,a.data,a.runtime);store.arm='M1_SCALED_DGM'
    rs=read(a.phase1/'D1_METRICS_PRIVATE.pkl');selects=[];histories=[];errors=[];audits=[];fr=json_load(a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json')
    for arm in ARMS:
        print('START_DECISION_MODULE',arm,flush=True)
        for fold in range(1,6):
            seed=42+1009*fold;cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';r=next(q for q in fr['selections'] if q['arm']=='D1' and q['fold']==fold);assert sha(cp)==r['checkpoint_sha256'];bp=cp.parent.parent/'BANK_PRIVATE.pt';assert sha(bp)==r['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False)
            sq=read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl');assert set(sq['fit_patients'])==set(d['patient'][d['train']]);scale=torch.tensor(sq['scale'],device='cuda')
            if fold==1:
                model=make(core,fac,records,cp,seed,arm);audits.append(admission(model,d,store,scale,core,arm));write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS_SO_FAR',candidates=audits));del model
            model=make(core,fac,records,cp,seed,arm);initial=core.state_hash(model.state_dict());best,h=fit(model,d,store,scale,a.runtime/arm/f'fold{fold}/TRAIN',seed,binding,core,a.runtime,arm,fold)
            cpp=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';pb=dict(binding=binding,checkpoint_sha256=sha(cpp),threshold=best['threshold']);pp=a.runtime/'predictions_private'/f'{arm}_fold{fold}.pkl'
            if pp.exists():pred=read(pp);assert pred['binding']==pb
            else:pred=dict(binding=pb,modes={})
            modes=['full','residual0','shuffle','static','identity']+[f'neutral_{k}' for k in range(7)]+(['margin_identity'] if arm=='M7_MARGIN_GATED' else [])
            for mode in modes:
                if mode not in pred['modes']:pred['modes'][mode]=EV.predict(model,d,d['val'],store,scale,mode);save(pp,pred)
                score=torch.sigmoid(torch.from_numpy(pred['modes'][mode]['logit'])).numpy();rs.extend(m.patient_rows(c,d,score,best['threshold'],arm if mode=='full' else arm+'__'+mode,fold))
            assert np.array_equal(pred['modes']['full']['base_logit'],pred['modes']['residual0']['logit'])
            reference=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl');assert np.max(np.abs(torch.sigmoid(torch.from_numpy(pred['modes']['full']['base_logit'])).numpy()-reference['score']))<=1e-6
            for mode in modes[1:]:errors.append(dict(arm=arm,**EV.errors(d,pred['modes']['full'],pred['modes'][mode],best['threshold'],fold,mode)))
            row=dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],epochs=len(h),initial_state_sha256=initial,checkpoint_sha256=sha(cpp),D1_checkpoint_sha256=sha(cp),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
            if arm=='M7_MARGIN_GATED':row.update(margin_a=float(model.margin_a.detach().cpu()),margin_b=float(model.margin_b.detach().cpu()))
            selects.append(row);histories.extend(h);save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rs);pd.DataFrame(selects).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False);print('DECISION_MODULE_FOLD_COMPLETE',arm,fold,flush=True);del model,d;gc.collect()
    write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS',candidates=audits));write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selects))
    report=load('decision_module_report',ROOT/'code/report.py');report.finish(a.runtime,a.phase2,c,rs,selects,errors,binding)

def json_load(p):return json.loads(Path(p).read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--phase2',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase2_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args()
    try:main(a)
    except Exception as e:
        write(a.runtime/'results/RUN_STATUS.json',dict(status='ENGINEERING_OR_INTEGRITY_FAILURE',exception_type=type(e).__name__,preserve_artifacts=True,no_new_Outer=True));raise
