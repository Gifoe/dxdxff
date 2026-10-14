"""Fixed 15-cell patient-held-out decision-adapter Development matrix."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bank import arguments,input_binding,read,save,write,sha,context
from models import ARMS,Adapter,loss
from explore import original_modules

def fit(a,arm,fold,d,core,binding):
    seed=42+1009*fold;core.seed_all(seed);model=Adapter(arm).cuda();cell=a.runtime/arm/f'fold{fold}/TRAIN';cell.mkdir(parents=True,exist_ok=True)
    x=torch.tensor(d['x'],device='cuda');base=torch.tensor(d['base'],device='cuda');y=torch.tensor(d['y'],dtype=torch.float32,device='cuda');tr=d['train'];va=d['val'];groups=[tr[k] for k in core.groups(d['patient'][tr])]
    pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1);bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    initial=core.state_hash(model.state_dict());bind=dict(input=binding,bank=sha(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl'),arm=arm,fold=fold,seed=seed,initial=initial)
    with torch.inference_mode():assert torch.equal(model(x,base),base)
    best=key=None;stale=0;history=[];start=1;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind;model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng']);best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
        if done.exists():
            cc=json.loads(done.read_text());assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last);model.load_state_dict(best['model']);return model,best,history
        if q['epoch']>=10 and stale>=10:start=61
    for epoch in range(start,61):
        tick=time.perf_counter();model.train();values=[]
        for n in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(n)];opt.zero_grad(set_to_none=True);z=model(x[ii],base[ii]);value=loss(z,y[ii],bce,arm);assert torch.isfinite(value);value.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();values.append(float(value.detach().cpu()))
        model.eval()
        with torch.inference_mode():s=model(x[va],base[va]).sigmoid().cpu().numpy()
        sel=core.select_threshold(d['y'][va],s,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(arm=arm,fold=fold,epoch=epoch,loss=float(np.mean(values)),seconds=time.perf_counter()-tick,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch));write(a.runtime/'results/RUN_STATUS.json',dict(status='DECISION_ADAPTER_TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],no_new_Outer=True));print('ADAPTER_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=10 and stale>=10:break
    assert best is not None;core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),threshold=best['threshold'],epoch=best['epoch']));model.load_state_dict(best['model']);return model,best,history

def main(a):
    binding=input_binding(a);assert json.loads((a.runtime/'INPUT_FREEZE_PRIVATE.json').read_text())==binding;assert json.loads((a.runtime/'results/EVIDENCE_BANK_AUDIT.json').read_text())['status']=='PASS'
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,b=original_modules();rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');sels=[];history=[]
    for fold in range(1,6):
        d=read(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl');s=torch.sigmoid(torch.from_numpy(d['base'][d['val']])).numpy();rows.extend(m.patient_rows(c,d,s,d['M8_threshold'],'M8_MACRO_AWARE',fold))
    for arm in ARMS:
        for fold in range(1,6):
            d=read(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl');model,best,h=fit(a,arm,fold,d,core,binding);model.eval()
            with torch.inference_mode():ll=model(torch.tensor(d['x'][d['val']],device='cuda'),torch.tensor(d['base'][d['val']],device='cuda')).cpu().numpy()
            cp=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';q=dict(binding=binding,checkpoint_sha256=sha(cp),logit=ll,threshold=best['threshold'],indices=d['val']);save(a.runtime/f'predictions_private/{arm}_fold{fold}.pkl',q)
            rows.extend(m.patient_rows(c,d,torch.sigmoid(torch.from_numpy(ll)).numpy(),best['threshold'],arm,fold));sels.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],epochs=len(h),parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(cp),bank_sha256=sha(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl')));history.extend(h)
            save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);pd.DataFrame(sels).to_csv(a.runtime/'results/MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(a.runtime/'results/TRAINING_HISTORY.csv',index=False);print('ADAPTER_CELL_COMPLETE',arm,fold,flush=True);del model;torch.cuda.empty_cache()
    df=pd.DataFrame(rows);means=df.groupby('arm')[m.METRICS].mean()
    for arm,g in df.groupby('arm'):assert len(g)==65 and g.patient.nunique()==47
    means.reset_index().to_csv(a.runtime/'results/DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:
        pd.DataFrame([dict(arm=arm,**{field:v},appearances=len(g),unique_IDs=g.patient.nunique(),**g[m.METRICS].mean().to_dict()) for (arm,v),g in df.groupby(['arm',field])]).to_csv(a.runtime/f'results/DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    boot=[]
    for arm in ARMS:
        for ref in ['D1','M8_MACRO_AWARE']:boot.extend(dict(comparison=f'{arm} minus {ref}',**r) for r in b.compare(c,df[df.arm==arm],df[df.arm==ref],m.METRICS))
    pd.DataFrame(boot).to_csv(a.runtime/'results/PAIRED_BOOTSTRAP.csv',index=False)
    chosen=means.loc[ARMS].sort_values(['macro_f1','ez_f1'],ascending=False).index[0];score=float(means.loc[chosen,'macro_f1']);quality=all(means.loc[chosen,k]>=means.loc['D1',k] for k in ['ez_auroc','ez_auprc','ez_f1'])
    gate=dict(status='ALL15_DEVELOPMENT_CELLS_COMPLETE',selected_common_candidate=chosen,MacroF1=score,target_strictly_over_070=score>.70,AUROC_AP_EZF1_vs_D1_quality=bool(quality),no_new_Outer=True,adaptive_repeated_Development=True,not_independent_confirmation=True)
    write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=sels,selected=chosen));write(a.runtime/'results/SCIENTIFIC_GATE.json',gate);write(a.runtime/'results/RUN_STATUS.json',gate);print('PHASE4_COMPLETE',json.dumps(gate),flush=True)

if __name__=='__main__':main(arguments(argparse.ArgumentParser()).parse_args())
