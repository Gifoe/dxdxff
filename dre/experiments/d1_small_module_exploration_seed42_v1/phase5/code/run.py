"""Two fixed joint-readout warm-start candidates; FIT/VAL only."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'tools'))
from explore import context,make_base,Inputs,EV,read,save,write,sha,original_modules
from best_recipe import build
from joint_model import Joint,ARMS,objective

def fit(a,arm,fold,model,d,store,scale,core,binding,initial_threshold):
    seed=42+1009*fold;core.seed_all(seed);cell=a.runtime/arm/f'fold{fold}/TRAIN';cell.mkdir(parents=True,exist_ok=True);params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0001,weight_decay=.001)
    tr,va=d['train'],d['val'];pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'));bind=dict(input=binding,arm=arm,fold=fold,initial=core.state_hash(model.state_dict()),D1=core.state_hash(model.base.state_dict()))
    last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json';best=key=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind;model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng']);best=q['best'];key=q['key'];history=q['history'];stale=q['stale'];start=q['epoch']+1
        if done.exists():
            cc=json.loads(done.read_text());assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last);model.load_state_dict(best['model']);return best,history
        if q['epoch']>=6 and stale>=6:start=21
    else:
        q=EV.predict(model,d,va,store,scale);s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();sel=core.select_threshold(d['y'][va],s,d['patient'][va]);key=(sel['macro_f1'],sel['ez_f1'],0);best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=0,**sel)
        history.append(dict(arm=arm,fold=fold,epoch=0,loss=None,seconds=0,**sel))
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,21):
        tick=time.perf_counter();model.train();values=[]
        for n in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(n)];pid=str(d['patient'][ii[0]]);raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.tensor(d['x'][ii],device='cuda');y=torch.tensor(d['y'][ii],dtype=torch.float32,device='cuda');opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);value=objective(q[0],y,bce);assert torch.isfinite(value);value.backward();assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in params if p.grad is not None);torch.nn.utils.clip_grad_norm_(params,1);opt.step();values.append(float(value.detach().cpu()));del raw,mask,times,delta,perms,x,y,q,value
        assert core.state_hash(model.base.state_dict())==bind['D1'] and not model.base.training;q=EV.predict(model,d,va,store,scale);s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();sel=core.select_threshold(d['y'][va],s,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(arm=arm,fold=fold,epoch=epoch,loss=float(np.mean(values)),seconds=time.perf_counter()-tick,cap=float(3*model.cap_parameter.detach().sigmoid().cpu()) if arm==ARMS[1] else 1,**sel))
        if candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch));write(a.runtime/'results/RUN_STATUS.json',dict(status='JOINT_TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],no_new_Outer=True));print('JOINT_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last)));model.load_state_dict(best['model']);return best,history

def main(a):
    a.runtime.mkdir(parents=True,exist_ok=True);pub=a.runtime/'results';pub.mkdir(exist_ok=True)
    previous=json.loads((a.previous/'INPUT_FREEZE_PRIVATE.json').read_text());assert previous['code']=={p.name:sha(p) for p in (ROOT/'phase3/code').glob('*.py')}
    assert json.loads((a.phase4/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
    binding=dict(protocol=sha(ROOT/'phase5/PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},previous_input=sha(a.previous/'INPUT_FREEZE_PRIVATE.json'),previous_models=sha(a.previous/'MODEL_FREEZE_PRIVATE.json'),phase4_verification=sha(a.phase4/'results/INDEPENDENT_VERIFICATION.json'),scales={str(f):sha(a.phase1/f'scale_fold{f}_PRIVATE.pkl') for f in range(1,6)})
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,b=original_modules();store=Inputs(a.baseline,a.data,a.previous);store.arm='M1_SCALED_DGM';fr=json.loads((a.previous/'MODEL_FREEZE_PRIVATE.json').read_text());rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');selections=[];history=[];audits=[]
    for arm in ARMS:
        for fold in range(1,6):
            cp=a.previous/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt';r=next(q for q in fr['selections'] if q['arm']=='M8_MACRO_AWARE' and q['fold']==fold);assert sha(cp)==r['checkpoint_sha256'];state=torch.load(cp,map_location='cpu',weights_only=False)['model'];bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);sq=read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl');assert set(sq['fit_patients'])==set(d['patient'][d['train']]);scale=torch.tensor(sq['scale'],device='cuda');base=make_base(core,fac,records,a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold);model=Joint(base,state,arm).cuda().eval()
            pid=sorted(set(d['patient'][d['train']]))[0];ii=d['train'][d['patient'][d['train']]==pid];raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.tensor(d['x'][ii],device='cuda');control=build(make_base(core,fac,records,a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold)).cuda().eval();control.load_state_dict(state,strict=True)
            with torch.inference_mode():drift=float((model(x,raw,mask,times,delta,scale)[0]-control(x,raw,mask,times,delta,scale)[0]).abs().max().cpu())
            assert drift<=1e-6;audits.append(dict(arm=arm,fold=fold,initial_M8_max_logit_difference=drift,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)));write(pub/'INITIAL_IDENTITY_AUDIT.json',dict(folds=audits));del control,raw,mask,times,delta,perms,x
            best,h=fit(a,arm,fold,model,d,store,scale,core,binding,r['threshold']);q=EV.predict(model,d,d['val'],store,scale);target=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';save(a.runtime/f'predictions_private/{arm}_fold{fold}.pkl',dict(binding=binding,checkpoint_sha256=sha(target),threshold=best['threshold'],**q));rows.extend(m.patient_rows(c,d,torch.sigmoid(torch.from_numpy(q['logit'])).numpy(),best['threshold'],arm,fold));selections.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],checkpoint_sha256=sha(target),M8_checkpoint_sha256=sha(cp),D1_bank_sha256=sha(bp)));history.extend(h);save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);pd.DataFrame(selections).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(pub/'TRAINING_HISTORY.csv',index=False);print('JOINT_CELL_COMPLETE',arm,fold,flush=True);del model,d;store.raw.cache.clear();gc.collect();torch.cuda.empty_cache()
    df=pd.DataFrame(rows);means=df.groupby('arm')[m.METRICS].mean();means.reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:df.groupby(['arm',field])[m.METRICS].mean().reset_index().to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    boots=[]
    for arm in ARMS:boots.extend(dict(comparison=arm+' minus D1',**q) for q in b.compare(c,df[df.arm==arm],df[df.arm=='D1'],m.METRICS))
    pd.DataFrame(boots).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);chosen=means.loc[ARMS].sort_values(['macro_f1','ez_f1'],ascending=False).index[0];write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections,selected=chosen));write(pub/'RUN_STATUS.json',dict(status='ALL10_JOINT_DEVELOPMENT_CELLS_COMPLETE',selected=chosen,MacroF1=float(means.loc[chosen,'macro_f1']),target_over_070=bool(means.loc[chosen,'macro_f1']>.7),no_new_Outer=True));print('JOINT_PHASE5_COMPLETE',chosen,float(means.loc[chosen,'macro_f1']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase5_runtime'));p.add_argument('--previous',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase4',type=Path,default=Path(r'D:\d1_macro70_phase4_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
