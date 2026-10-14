"""Fixed three-candidate exploration; new prospective lock, no Outer loader."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parents[1]
DGM_CODE=Path(__import__('os').environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'))
sys.path.insert(0,str(DGM_CODE))
from runtime_support import BASE_CODE,sha,read,save,write,load,context,Store,make_base,original_modules
from dgm_net import DGMNet
from exploration_models import DeltaResidual
EV=load('small_module_frozen_evaluate',DGM_CODE/'evaluate.py')

ARMS=['M1_SCALED_DGM','M2_STAGE_DELTA','M3_DELTA_SET']

class Inputs:
    def __init__(self,baseline,data,runtime):
        self.raw=Store(baseline);self.registry=read(data/'CONDITIONING_REGISTRY_PRIVATE.pkl');self.cache={};self.arm=None
    def get(self,patient,channels,device):
        pid=str(patient);r=self.registry['files'][pid]
        if pid not in self.cache:
            assert sha(Path(r['path']))==r['sha256'];q=read(Path(r['path']));plan=self.raw.registry['plans'][pid]
            assert sha(Path(plan['path']))==q['raw_bank_sha256']==plan['sha256']
            b=torch.load(plan['path'],map_location='cpu',weights_only=False)
            assert q['channels']==b['channels'] and pid==q['patient']==b['patient']
            self.cache[pid]=dict(delta=q['delta'],perms=q['perms'],channels=q['channels'],mask=b['mask'][0],times=torch.tensor(plan['times'],dtype=torch.float32))
            del b
        q=self.cache[pid];ix=[q['channels'].index(c) for c in channels]
        if self.arm=='M1_SCALED_DGM':raw,mask,times,_=self.raw.get(pid,channels,device)
        else:
            raw=torch.empty(0,device=device);mask=q['mask'].index_select(1,torch.tensor(ix)).to(device);times=q['times'].to(device)
        return raw,mask,times,torch.from_numpy(q['delta'][:,ix]).to(device),[p[ix] for p in q['perms']]

def make(core,fac,records,cp,seed,arm):
    base=make_base(core,fac,records,cp,seed)
    return (DGMNet(base) if arm=='M1_SCALED_DGM' else DeltaResidual(base,arm)).cuda()

def admission(model,d,store,scale,core,arm):
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected={'M1_SCALED_DGM':9482,'M2_STAGE_DELTA':722,'M3_DELTA_SET':1354}[arm]
    assert trainable==expected and total==trainable+1537
    tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
    model.eval(); frozen=core.state_hash(model.base.state_dict())
    with torch.inference_mode():q=model(x,raw,mask,times,delta,scale);assert torch.equal(q[0],q[2])
    pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
    norms=[]
    for step in range(2):
        model.train();opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=fn(q[0],y)+.25*fn(q[1],y);loss.backward()
        assert all(p.grad is None for p in model.base.parameters()) and not model.base.training
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ns={n:float(p.grad.norm().cpu()) for n,p in model.named_parameters() if p.grad is not None};assert ns['alpha']>0 and ns['head.0.weight']>0
        if arm=='M1_SCALED_DGM':
            assert ns['gate.network.2.weight']>0
            if step:assert ns['gate.network.0.weight']>0
        if arm=='M3_DELTA_SET':assert ns['encoder.0.weight']>0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();norms.append(dict(step=step+1,norms=ns))
    assert core.state_hash(model.base.state_dict())==frozen
    return dict(arm=arm,status='PASS',total=total,trainable=trainable,frozen_D1=1537,initial_D1_identity_exact=True,FIT_two_step_smoke_discarded=True,gradients=norms)

def fit(model,d,store,scale,cell,seed,binding,core,runtime,arm,fold):
    cell.mkdir(parents=True,exist_ok=True);bind=dict(inputs=binding,arm=arm,fold=fold,seed=seed,initial=core.state_hash(model.state_dict()),D1=core.state_hash(model.base.state_dict()),scale=scale.cpu().tolist())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);core.seed_all(seed);tr,va=d['train'],d['val']
    pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'))
    best=key=None;history=[];stale=0;start=1;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind;model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng']);best=q['best'];key=q['key'];history=q['history'];stale=q['stale'];start=q['epoch']+1
        if done.exists():
            cc=json.loads(done.read_text());assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last);model.load_state_dict(best['model']);return best,history
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,31):
        t=time.perf_counter();model.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];pid=str(d['patient'][ii[0]]);raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale);loss=fn(q[0],y)+.25*fn(q[1],y);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None for p in model.base.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach().cpu()));del raw,mask,times,delta,perms,x,y,q,loss
        assert core.state_hash(model.base.state_dict())==bind['D1'] and not model.base.training
        pred=EV.predict(model,d,va,store,scale);score=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy();sel=core.select_threshold(d['y'][va],score,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(arm=arm,fold=fold,epoch=epoch,loss=float(np.mean(losses)),alpha=float(model.alpha.detach().cpu()),seconds=time.perf_counter()-t,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(runtime/'results/RUN_STATUS.json',dict(status='EXPLORATION_TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],no_new_Outer=True))
        print('SMALL_MODULE_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None;core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),threshold=best['threshold'],epoch=best['epoch']))
    model.load_state_dict(best['model']);return best,history

def run(a):
    pub=a.runtime/'results';pub.mkdir(parents=True,exist_ok=True);lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());assert a.arms==lock['models'], 'Phase1 fixed full matrix only'
    old=json.loads((a.data/'results/WINDOW_DELTA_PARITY.json').read_text());assert old['status']=='PASS' and old['maximum_actual_7D_reaggregation_error']==0
    assert all(sha(DGM_CODE/f)==h for f,h in old['binding']['code'].items());assert sha(BASE_CODE.parent/'PROTOCOL_LOCK.json')==old['binding']['baseline_protocol']
    paths=dict(raw=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'),window=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))
    hashes={k:sha(p) for k,p in paths.items()};assert hashes==old['binding']['inputs']
    binding=dict(protocol=sha(ROOT/'PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},DGM_data_gate=sha(a.data/'results/WINDOW_DELTA_PARITY.json'),DGM_code=old['binding']['code'],input_hashes=hashes)
    frozen=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if frozen.exists():assert json.loads(frozen.read_text())==binding
    else:write(frozen,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);metrics,boot=original_modules();store=Inputs(a.baseline,a.data,a.runtime)
    rs=[];selections=[];histories=[];err=[];audits=[];scale_audit=[];D1parity=[]
    fr=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text())
    for fold in range(1,6):
        cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'; r=next(q for q in fr['selections'] if q['arm']=='D1' and q['fold']==fold);assert sha(cp)==r['checkpoint_sha256'];bp=cp.parent.parent/'BANK_PRIVATE.pt';assert sha(bp)==r['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False)
        b=make_base(core,fac,records,cp,42+1009*fold).cuda().eval()
        with torch.inference_mode():s=torch.sigmoid(b(torch.as_tensor(d['x'][d['val']],device='cuda'))).cpu().numpy()
        original=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl');assert np.max(np.abs(s-original['score']))<=1e-6 and np.array_equal(s>=r['threshold'],original['score']>=original['threshold'])
        rs.extend(metrics.patient_rows(c,d,s,r['threshold'],'D1',fold));D1parity.append(dict(fold=fold,max_score_difference=float(np.max(np.abs(s-original['score']))),checkpoint_sha256=sha(cp),decisions_exact=True));del b
        oldscale=read(a.data/f'scale_fold{fold}_PRIVATE.pkl');assert set(oldscale['fit_patients'])==set(d['patient'][d['train']])
        typical=np.median(oldscale['patient_magnitudes'],0);scale=np.maximum(typical,1e-30).astype(np.float32)
        save(a.runtime/f'scale_fold{fold}_PRIVATE.pkl',dict(binding=binding,scale=scale,fit_patients=oldscale['fit_patients'],labels_used=False,source_sha256=sha(a.data/f'scale_fold{fold}_PRIVATE.pkl')))
        for scope,ids in [('FIT',sorted(set(d['patient'][d['train']]))),('Development',sorted(set(d['patient'][d['val']])) )]:
            clipped=[];magnitudes=[];store.arm='M2_STAGE_DELTA'
            for pid in ids:
                raw,mask,times,delta,perms=store.get(pid,store.registry['files'][str(pid)]['channels'],'cpu');v=delta[mask].numpy()/scale;clipped.append((np.abs(v)>8).mean(0));magnitudes.append(np.median(np.abs(v),0))
            for k,feature in enumerate(lock['models'][:0]+['beta','low_gamma','partial_high_gamma','rms','variance','line_length_sec','entropy']):scale_audit.append(dict(fold=fold,scope=scope,feature=feature,typical_abs_delta=float(typical[k]),scale=float(scale[k]),floor_active=bool(typical[k]<1e-30),patient_equal_clipped_fraction=float(np.mean(clipped,0)[k]),median_patient_abs_normalized=float(np.median(magnitudes,0)[k]),labels_used=False))
    write(pub/'D1_PARITY.json',dict(status='PASS',folds=D1parity));pd.DataFrame(scale_audit).to_csv(pub/'FEATURE_SCALE_AUDIT.csv',index=False)
    # Save original replay before training and never open any Outer score artifact.
    save(a.runtime/'D1_METRICS_PRIVATE.pkl',rs)
    for arm in ARMS:
        store.arm=arm; print('START_CANDIDATE',arm,flush=True)
        for fold in range(1,6):
            seed=42+1009*fold;cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';d=torch.load(cp.parent.parent/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False);scale=torch.tensor(read(a.runtime/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            if fold==1:
                m=make(core,fac,records,cp,seed,arm);audit=admission(m,d,store,scale,core,arm);audits.append(audit);del m;write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS_SO_FAR',candidates=audits))
            model=make(core,fac,records,cp,seed,arm);initial=core.state_hash(model.state_dict());best,h=fit(model,d,store,scale,a.runtime/arm/f'fold{fold}/TRAIN',seed,binding,core,a.runtime,arm,fold)
            checkpoint=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';pb=dict(input=binding,checkpoint=sha(checkpoint),threshold=best['threshold']);pp=a.runtime/'predictions_private'/f'{arm}_fold{fold}.pkl'
            if pp.exists():pred=read(pp);assert pred['binding']==pb
            else:pred=dict(binding=pb,modes={})
            modes=['full','residual0','shuffle','static']+[f'neutral_{k}' for k in range(7)]+(['identity'] if arm=='M1_SCALED_DGM' else [])
            for mode in modes:
                if mode not in pred['modes']:pred['modes'][mode]=EV.predict(model,d,d['val'],store,scale,mode);save(pp,pred)
                q=pred['modes'][mode];score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();rs.extend(metrics.patient_rows(c,d,score,best['threshold'],arm if mode=='full' else arm+'__'+mode,fold))
            assert np.array_equal(pred['modes']['residual0']['logit'],pred['modes']['full']['base_logit'])
            original=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl');assert np.max(np.abs(torch.sigmoid(torch.from_numpy(pred['modes']['full']['base_logit'])).numpy()-original['score']))<=1e-6
            for mode in modes[1:]:err.append(dict(arm=arm,**EV.errors(d,pred['modes']['full'],pred['modes'][mode],best['threshold'],fold,mode)))
            selections.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],epochs=len(h),checkpoint_sha256=sha(checkpoint),D1_checkpoint_sha256=sha(cp),initial_state_sha256=initial,alpha=float(model.alpha.detach().cpu()),residual_coefficient=float(.5*torch.tanh(model.alpha.detach()).cpu()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            histories.extend(h);save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rs);pd.DataFrame(selections).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False);print('CANDIDATE_FOLD_COMPLETE',arm,fold,flush=True);del model,d;gc.collect()
        store.raw.cache.clear();gc.collect()
    write(pub/'PARAMETER_AND_GRADIENT_AUDIT.json',dict(status='PASS',candidates=audits));write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections))
    report=load('small_module_report',ROOT/'code/report.py');report.finish(a.runtime,c,rs,selections,err,binding)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--arms',nargs='+',default=ARMS);a=p.parse_args()
    try:run(a)
    except Exception as e:
        write(a.runtime/'results/RUN_STATUS.json',dict(status='ENGINEERING_OR_INTEGRITY_FAILURE',type=type(e).__name__,artifacts_preserved=True,no_new_Outer=True));raise
