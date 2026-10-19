"""Exactly matched full-patient training;15 verified reuses and45 fresh fits."""
import os
import re
import time
from common import *

def reuse(seed,fold,d,scale,core,fac,records):
    r=previous_cell(seed,fold);v=read(r['calibration']);m=make(core,fac,records,fold,'FULL',seed)
    assert state_hash(m.state_dict())==v['binding']['initial'] and preprocessor(d,scale,'FULL')==r['preprocessor']
    assert any(x['seed']==str(seed) and x['fold']==str(fold) and x['checkpoint_sha256']==r['checkpoint_sha256'] for x in csv.DictReader((PUB/'ADMISSION_REUSE.csv').open(newline='',encoding='utf-8')))
    return dict(r,arm='FULL',reused=True,previous_arm='B0'),v['history']

def fit(arm,seed,fold,binding,core,fac,records,d,scale,inputs):
    cell=ROOT/'models_private'/arm/f'seed{seed}'/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    m=make(core,fac,records,fold,arm,seed)
    bind=dict(input=binding,arm=arm,seed=seed,fold=fold,initial=state_hash(m.state_dict()),preprocessor=preprocessor(d,scale,arm))
    last,bestfile,calfile,done=[cell/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','CALIBRATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and q['best']==sha(bestfile) and q['calibration']==sha(calfile)
        v=read(calfile);assert v['binding']==bind;return v['summary'],v['history']
    s=seed+1009*fold;seed_all(s);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
    tr,va=d['train'],d['val'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    best=key=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,weights_only=False,map_location='cpu');assert q['binding']==bind
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng'])
        best,key,history,stale,start=q['best'],q['key'],q['history'],q['stale'],q['epoch']+1
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    for epoch in range(start,31):
        tick=time.perf_counter();m.train();losses=[]
        for k in np.random.default_rng(s+epoch).permutation(len(groups)):
            ii=groups[int(k)];y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);out,_=forward(m,d,ii,inputs,scale);loss=MODEL.objective(out,y,pos)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()));del out,loss,y
        logits,_=predict(m,d,va,inputs,scale);sel=core.select_threshold(d['y'][va],probability(logits),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(arm=arm,seed=seed,fold=fold,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**sel))
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'PROGRESS.json',dict(status='TRAINING',arm=arm,seed=seed,fold=fold,epoch=epoch,best_epoch=best['epoch']))
        print('PRF_EPOCH',arm,seed,fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=6 and stale>=6:break
    m.load_state_dict(best['model']);logits,_=predict(m,d,va,inputs,scale)
    selected=core.select_threshold(d['y'][va],probability(logits),d['patient'][va])
    assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in selected.items())
    atomic_torch(bestfile,dict(binding=bind,**best))
    summary=dict(arm=arm,seed=seed,fold=fold,epochs=len(history),epoch=int(best['epoch']),threshold=float(best['threshold']),parameters=11425,
                 checkpoint=str(bestfile),checkpoint_sha256=sha(bestfile),calibration=str(calfile),preprocessor=preprocessor(d,scale,arm),reused=False)
    save(calfile,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=logits,summary=summary,history=history))
    write(done,dict(binding=digest(bind),best=sha(bestfile),calibration=sha(calfile)))
    del m,opt;gc.collect();torch.cuda.empty_cache();return summary,history

def main():
    binding,core,fac,records,rec,banks=setup();assert jr(PUB/'ADMISSION.json')['status']=='PASS'
    assert jr(PUB/'ADMISSION.json')['source']==sources()
    commit=os.environ.get('PRF_SOURCE_COMMIT','');assert re.fullmatch('[0-9a-f]{40}',commit),'Verified pre-training source push required'
    freeze(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json',dict(commit=commit,source=sources(),source_pushed_before_formal_fits=True))
    inputs=store();cells=[];histories=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        for seed in SEEDS:
            for arm in ARMS:
                if arm=='FULL':r,h=reuse(seed,fold,d,scale,core,fac,records)
                else:r,h=fit(arm,seed,fold,binding,core,fac,records,d,scale,inputs)
                cells.append(dict(r,calibration_sha256=sha(r['calibration'])))
                for row in h:
                    histories.append(dict(arm=arm,seed=seed,fold=fold,reused=arm=='FULL',**{k:row[k] for k in ['epoch','seconds','loss','threshold','macro_f1','ez_f1']}))
                csv_out('TRAIN_CELLS',[{k:q[k] for k in ['arm','seed','fold','epochs','epoch','threshold','parameters','checkpoint_sha256','calibration_sha256','reused']} for q in cells])
                csv_out('TRAIN_HISTORY',histories);print('PRF_TRAIN_CELL_COMPLETE',arm,seed,fold,len(cells),60,flush=True)
    assert {(r['arm'],r['seed'],r['fold']) for r in cells}==cells_expected()
    freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL60_COMPLETE',fresh_fits=45,reused=15,new_epochs=sum(not r['reused'] for r in histories),reused_epochs=sum(r['reused'] for r in histories)))
    print('PRF_ALL60_SELECTIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
