"""Only A2, original five Development folds; exact epoch checkpoint resume."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from runtime_support import BASE_CODE,sha,read,save,write,load,context,Store,make_base,original_modules
from dgm_net import DGMNet

OWN=Path(__file__).parent
ev=load('dgm_development_evaluator',OWN/'evaluate.py')
vv=load('dgm_engineering_verify',OWN/'verify.py')

class ConditioningStore:
    def __init__(self, baseline, runtime):
        self.raw=Store(baseline);self.reg=read(runtime/'CONDITIONING_REGISTRY_PRIVATE.pkl');self.cache={}
    def get(self,patient,channels,device):
        pid=str(patient);r=self.reg['files'][pid]
        if pid not in self.cache:
            assert sha(Path(r['path']))==r['sha256'];self.cache[pid]=read(Path(r['path']))
        q=self.cache[pid]; raw,mask,times,_=self.raw.get(pid,channels,device)
        ix=[q['channels'].index(ch) for ch in channels]
        delta=torch.from_numpy(q['delta'][:,ix]).to(device); perms=[p[ix] for p in q['perms']]
        return raw,mask,times,delta,perms

def make(core,fac,records,cp,seed):
    return DGMNet(make_base(core,fac,records,cp,seed)).cuda()

def fit(model,d,store,scale,cell,seed,binding,core,runtime,fold):
    cell.mkdir(parents=True,exist_ok=True)
    bind=dict(input=binding,initial=core.state_hash(model.state_dict()),frozen_D1=core.state_hash(model.base.state_dict()),seed=seed,scale=scale.detach().cpu().tolist())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);core.seed_all(seed)
    tr,va=d['train'],d['val'];pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1)
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda')); history=[];best=None;key=None;stale=0;start=1
    last=cell/'LAST_PRIVATE.pt';complete=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        ck=torch.load(last,map_location='cuda',weights_only=False);assert ck['binding']==bind
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);core.restore_rng(ck['rng']);history=ck['history'];best=ck['best'];key=ck['key'];stale=ck['stale'];start=ck['epoch']+1
        if complete.exists():
            done=json.loads(complete.read_text());assert done['binding']==bind and done['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and done['last_sha256']==sha(last)
            model.load_state_dict(best['model']);return best,history
        if ck['epoch']>=6 and stale>=6:start=31
        print('A2_EXACT_EPOCH_RESUME',fold,start,flush=True)
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,31):
        t=time.perf_counter();model.train();losses=[];auxlosses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];pid=str(d['patient'][ii[0]])
            raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
            x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);ll,rr,base,res,g=model(x,raw,mask,times,delta,scale)
            aux=fn(rr,y);loss=fn(ll,y)+.25*aux;assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None for p in model.base.parameters())
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            opt.step();losses.append(float(loss.detach().cpu()));auxlosses.append(float(aux.detach().cpu()))
            del raw,mask,times,delta,perms,x,y,ll,rr,base,res,g,aux,loss
        assert not model.base.training and core.state_hash(model.base.state_dict())==bind['frozen_D1']
        pred=ev.predict(model,d,va,store,scale);score=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy()
        sel=core.select_threshold(d['y'][va],score,d['patient'][va]); candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),auxiliary_loss=float(np.mean(auxlosses)),alpha=float(model.alpha.detach().cpu()),effective_residual_coefficient=float(.5*torch.tanh(model.alpha.detach()).cpu()),wall_seconds=time.perf_counter()-t,**sel))
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),history=history,best=best,key=key,stale=stale,epoch=epoch))
        write(runtime/'results/RUN_STATUS.json',dict(status='A2_DEVELOPMENT_TRAINING',fold=fold,epoch=epoch,best_epoch=best['epoch'],folds_complete=fold-1,total_folds=5,seconds_last_epoch=history[-1]['wall_seconds'],Outer_accessed=False))
        print('A2_EPOCH',fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['wall_seconds'],2),flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None
    core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(complete,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),epoch=best['epoch'],threshold=best['threshold']))
    model.load_state_dict(best['model']);model.eval();return best,history

def main(a):
    root=OWN.parent; pub=a.runtime/'results'; data=json.loads((pub/'WINDOW_DELTA_PARITY.json').read_text());assert data['status']=='PASS'
    input_binding=json.loads((a.runtime/'INPUT_FREEZE_PRIVATE.json').read_text()); assert input_binding==data['binding']
    assert input_binding['protocol']==sha(root/'PROTOCOL_LOCK.json') and input_binding['code']=={p.name:sha(p) for p in OWN.glob('*.py')}
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline); metrics,boot=original_modules();store=ConditioningStore(a.baseline,a.runtime)
    binding=dict(inputs=sha(a.runtime/'INPUT_FREEZE_PRIVATE.json'),conditioning=sha(a.runtime/'CONDITIONING_REGISTRY_PRIVATE.pkl'),dependencies=deps)
    allrows=read(a.runtime/'A0_DEVELOPMENT_PRIVATE.pkl');selects=[];histories=[];errors=[];activities=[];params=[]
    freeze=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text())
    # Actual legal FIT gradient test. Discard and reset all model/optimizer/RNG before training.
    if not (pub/'PARAMETER_AND_GRADIENT_AUDIT.json').exists():
        cp=a.baseline/'D1/fold1/TRAIN/BEST_PRIVATE.pt';d=torch.load(cp.parent.parent/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        scale=torch.tensor(read(a.runtime/'scale_fold1_PRIVATE.pkl')['scale'],device='cuda')
        model=make(core,fac,records,cp,1051);audit=vv.check_model(model,d,store,scale,core);write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',audit);del model,d
        print('ACTUAL_FIT_GRADIENT_AND_IDENTITY_PASS',flush=True)
    else:assert json.loads((pub/'PARAMETER_AND_GRADIENT_AUDIT.json').read_text())['status']=='PASS'
    for fold in range(1,6):
        seed=42+1009*fold;cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';row=next(q for q in freeze['selections'] if q['arm']=='D1' and q['fold']==fold)
        assert sha(cp)==row['checkpoint_sha256'];bp=cp.parent.parent/'BANK_PRIVATE.pt'; assert sha(bp)==row['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False)
        sq=read(a.runtime/f'scale_fold{fold}_PRIVATE.pkl'); assert set(sq['fit_patients'])==set(d['patient'][d['train']])
        scale=torch.tensor(sq['scale'],device='cuda');model=make(core,fac,records,cp,seed);initial=core.state_hash(model.state_dict())
        best,h=fit(model,d,store,scale,a.runtime/f'A2/fold{fold}/TRAIN',seed,binding,core,a.runtime,fold)
        ck=a.runtime/f'A2/fold{fold}/TRAIN/BEST_PRIVATE.pt'; pbind=dict(input=binding,checkpoint=sha(ck),threshold=best['threshold'])
        pp=a.runtime/f'predictions_private/A2_fold{fold}.pkl'
        if pp.exists():
            pred=read(pp);assert pred['binding']==pbind
        else:
            pred=dict(binding=pbind,modes={})
        # Save each intervention atomically, including unfinished-fold safe resume.
        for mode in ev.MODES:
            if mode not in pred['modes']:
                pred['modes'][mode]=ev.predict(model,d,d['val'],store,scale,mode,diagnostics=True);save(pp,pred)
            q=pred['modes'][mode]; score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
            arm='A2' if mode=='full' else 'A2_'+mode
            allrows.extend(metrics.patient_rows(c,d,score,best['threshold'],arm,fold))
            activities.extend(dict(fold=fold,intervention=mode,**r) for r in q['gate_rows'])
        full=pred['modes']['full'];original=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl')
        embedded=torch.sigmoid(torch.from_numpy(full['base_logit'])).numpy(); assert np.max(np.abs(embedded-original['score']))<=1e-6
        assert np.array_equal(full['base_logit'],pred['modes']['residual0']['logit'])
        assert core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(full['logit'])).numpy(),d['patient'][d['val']])['threshold']==best['threshold']
        for mode in ev.MODES[1:]:errors.append(ev.errors(d,full,pred['modes'][mode],best['threshold'],fold,mode))
        auxscore=torch.sigmoid(torch.from_numpy(full['raw_logit'])).numpy();allrows.extend(metrics.patient_rows(c,d,auxscore,.5,'A2_auxiliary_diagnostic',fold))
        selects.append(dict(arm='A2',fold=fold,epoch=best['epoch'],threshold=best['threshold'],epochs=len(h),checkpoint_sha256=sha(ck),D1_checkpoint_sha256=sha(cp),initial_state_sha256=initial,alpha=float(model.alpha.detach().cpu()),effective_residual_coefficient=float(.5*torch.tanh(model.alpha.detach()).cpu())))
        histories.extend(dict(fold=fold,**r) for r in h);params.extend(dict(fold=fold,component=k,parameters=v) for k,v in dict(frozen_D1=1537,CNN=2160,gate=1048,head=6273,alpha=1,total=11019,trainable=9482).items())
        save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',allrows);pd.DataFrame(selects).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
        print('A2_FOLD_SELECTED_AND_INTERVENTIONS_COMPLETE',fold,best['epoch'],flush=True);del model,d,pred;gc.collect()
    write(a.runtime/'A2_DEVELOPMENT_FREEZE_PRIVATE.json',dict(binding=binding,selections=selects))
    report=load('dgm_reports',OWN/'report_a2.py');report.finish(a.runtime,c,allrows,selects,errors,activities,params)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except Exception as exc:
        write(a.runtime/'results/RUN_STATUS.json',dict(status='ENGINEERING_OR_INTEGRITY_FAILURE',exception_type=type(exc).__name__,preserve_all_artifacts=True,Outer_accessed=False))
        raise
