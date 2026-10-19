"""55 matched fresh fits; 15 exact previously selected reference cells."""
import os
import re
import time
from common import *

def fit(arm,fold,binding,core,fac,records,banks,inputs):
    d=bank(fold,banks,arm);ss=scale(fold);seed=42+1009*fold;m=make(core,fac,records,fold,arm)
    root=ROOT/'models_private'/arm/f'fold{fold}';root.mkdir(parents=True,exist_ok=True)
    bind=dict(input=binding,arm=arm,fold=fold,initial=state_hash(m.state_dict()),preprocessor=preprocessor(d,ss))
    last,bp,vp,done=[root/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','VALIDATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and sha(bp)==q['best_sha256'] and sha(vp)==q['validation_sha256']
        v=read(vp);assert v['binding']==bind;return v['summary'],v['history']
    seed_all(seed);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
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
        for k in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(k)];y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);logit,_=forward(m,d,ii,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del logit,loss,y
        out=predict(m,d,va,inputs,ss,fold,binding)
        sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(arm=arm,fold=fold,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**sel))
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],new_fits_total=55))
        print('NI_ABLATION_EPOCH',arm,fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=6 and stale>=6:break
    assert best['epoch']>=1;m.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best))
    out=predict(m,d,va,inputs,ss,fold,binding);sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va])
    assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in sel.items())
    summary=dict(arm=arm,fold=fold,epochs=len(history),epoch=int(best['epoch']),threshold=float(best['threshold']),parameters=sum(p.numel() for p in m.parameters()),
        checkpoint=str(bp),checkpoint_sha256=sha(bp),validation=str(vp),preprocessor=preprocessor(d,ss),reused=False)
    save(vp,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=out,summary=summary,history=history))
    write(done,dict(binding=digest(bind),best_sha256=sha(bp),validation_sha256=sha(vp)))
    del m,opt;gc.collect();torch.cuda.empty_cache();return summary,history

def main():
    binding,core,fac,records,rec,banks=setup();assert jr(PUB/'ADMISSION.json')['status']=='PASS'
    commit=os.environ.get('NI_SOURCE_COMMIT','');assert re.fullmatch('[0-9a-f]{40}',commit),'Verified pre-training source push required'
    freeze(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json',dict(commit=commit,source=sources(),source_pushed_before_formal_fits=True))
    inputs=store();cells=[];histories=[]
    for fold in range(1,6):
        for arm in ARMS:
            if arm in REF_NAMES:
                r,state,v=reference(arm,fold);m=make(core,fac,records,fold,arm)
                assert state_hash(m.state_dict())==v['binding']['initial']
                r=dict(r,epochs=0,parameters=sum(p.numel() for p in m.parameters()),preprocessor=preprocessor(bank(fold,banks,arm),scale(fold)));del m
            else:
                r,h=fit(arm,fold,binding,core,fac,records,banks,inputs);histories+=h;r=dict(r,validation_sha256=sha(r['validation']))
            cells.append(r)
            csv_out('TRAIN_CELLS',[{k:q[k] for k in ['arm','fold','epochs','epoch','threshold','parameters','checkpoint_sha256','validation_sha256','reused']} for q in cells])
            if histories:csv_out('TRAIN_HISTORY',histories)
            print('NI_TRAIN_CELL_COMPLETE',arm,fold,len(cells),70,flush=True)
    assert len(cells)==70 and sum(not r['reused'] for r in cells)==55
    freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL70_MODEL_CELLS_FROZEN',actual_new_fits=55,reused_cells=15,epochs=len(histories),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json')))
    print('NI_ALL55_NEW_FITS_PLUS15_REUSED_FROZEN',flush=True)

if __name__=='__main__':main()
