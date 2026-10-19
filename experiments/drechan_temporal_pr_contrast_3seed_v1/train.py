"""Original full-patient training; exact current-TRAIN global references for B2."""
import os
import re
import time
from common import *

def reference_final(fold,core,fac,records,d,scale,binding):
    fr=jr(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');r=dict(next(q for q in fr['cells'] if q['arm']=='FULL_F0' and q['fold']==fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');v=read(r['validation'])
    m=make(core,fac,records,fold,'B0',42);assert state_hash(m.state_dict())==v['binding']['initial']
    return dict(r,arm='B0',seed=42,stage='final',epochs=0,reused=True,calibration=r['validation'],calibration_sha256=r['validation_sha256'],
        preprocessor=preprocessor(d,scale),parameters=11425,reference_hash=None,reference_arm='FULL_F0'),[]

def fit(stage,arm,seed,fold,binding,core,fac,records,d,scale,inputs):
    cell=ROOT/'models_private'/stage/arm/f'seed{seed}'/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    m=make(core,fac,records,fold,arm,seed);initial=state_hash(m.state_dict())
    bind=dict(input=binding,stage=stage,arm=arm,seed=seed,fold=fold,initial=initial,preprocessor=preprocessor(d,scale))
    last,bestfile,calfile,done=[cell/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','CALIBRATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and q['best']==sha(bestfile) and q['calibration']==sha(calfile)
        v=read(calfile);assert v['binding']==bind;return v['summary'],v['history']
    globalbank=MODEL.GlobalReference(m,d,inputs,scale) if arm=='B2' else None
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
        tick=time.perf_counter();m.train();losses=[];refresh=globalbank.refreshes if globalbank else 0
        for k in np.random.default_rng(s+epoch).permutation(len(groups)):
            ii=groups[int(k)];y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            if globalbank:m.set_reference(globalbank.fit(m))
            opt.zero_grad(set_to_none=True);out,_=forward(m,d,ii,inputs,scale);loss=MODEL.objective(out,y,pos)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()));del out,loss,y
        if globalbank:m.set_reference(globalbank.fit(m));assert globalbank.refreshes-refresh==len(groups)+1
        logits,_=predict(m,d,va,inputs,scale);sel=core.select_threshold(d['y'][va],probability(logits),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},reference=reference_copy(m.reference),epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(stage=stage,arm=arm,seed=seed,fold=fold,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),global_reference_refreshes=(globalbank.refreshes-refresh if globalbank else 0),**sel))
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'PROGRESS.json',dict(status='TRAINING',stage=stage,arm=arm,seed=seed,fold=fold,epoch=epoch,best_epoch=best['epoch']))
        print('TPRC_EPOCH',stage,arm,seed,fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=6 and stale>=6:break
    m.load_state_dict(best['model']);m.set_reference(best['reference'])
    if globalbank:
        recalculated=globalbank.fit(m);assert reference_hash(recalculated)==reference_hash(m.reference)
    logits,_=predict(m,d,va,inputs,scale);selected=core.select_threshold(d['y'][va],probability(logits),d['patient'][va])
    assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in selected.items())
    atomic_torch(bestfile,dict(binding=bind,**best))
    summary=dict(stage=stage,arm=arm,seed=seed,fold=fold,epochs=len(history),epoch=int(best['epoch']),threshold=float(best['threshold']),parameters=sum(p.numel() for p in m.parameters()),
        checkpoint=str(bestfile),checkpoint_sha256=sha(bestfile),calibration=str(calfile),preprocessor=preprocessor(d,scale),reference_hash=reference_hash(best['reference']),reused=False)
    save(calfile,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=logits,summary=summary,history=history))
    write(done,dict(binding=digest(bind),best=sha(bestfile),calibration=sha(calfile)))
    del m,opt,globalbank;gc.collect();torch.cuda.empty_cache();return summary,history

def main(stage):
    assert stage in ['internal','final'];binding,core,fac,records,rec,banks=setup()
    assert jr(PUB/'ADMISSION.json')['status']=='PASS'
    commit=os.environ.get('TPRC_SOURCE_COMMIT','');assert re.fullmatch('[0-9a-f]{40}',commit),'Verified pre-training source push required'
    freeze(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json',dict(commit=commit,source=sources(),source_pushed_before_formal_fits=True))
    if stage=='final':assert jr(PUB/'internal/GATE.json')['continue_final'] and jr(PUB/'internal/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    inputs=store();cells=[];histories=[];roles=[]
    for fold in range(1,6):
        d,scale=data(fold,stage,core,rec,banks,inputs,binding)
        roles.append(dict(fold=fold,stage=stage,train_patients=len(d['roles']['train']),cal_patients=len(d['roles']['cal']),assessment_patients=len(d['roles']['assessment']),
            train_channels=len(d['train']),cal_channels=len(d['val']),assessment_channels=len(d['assessment']),preprocessor_hash=digest(preprocessor(d,scale))))
        for seed in SEEDS:
            for arm in ARMS:
                if stage=='final' and seed==42 and arm=='B0':r,h=reference_final(fold,core,fac,records,d,scale,binding)
                else:r,h=fit(stage,arm,seed,fold,binding,core,fac,records,d,scale,inputs)
                r=dict(r,calibration_sha256=sha(r['calibration']));cells.append(r);histories+=h
                csv_out(stage+'/TRAIN_CELLS',[{k:q[k] for k in ['stage','arm','seed','fold','epochs','epoch','threshold','parameters','checkpoint_sha256','calibration_sha256','reference_hash','reused']} for q in cells])
                if histories:csv_out(stage+'/TRAIN_HISTORY',histories)
                print('TPRC_TRAIN_CELL_COMPLETE',stage,arm,seed,fold,len(cells),60,flush=True)
    assert {(r['arm'],r['seed'],r['fold']) for r in cells}==cells_expected()
    csv_out(stage+'/ROLE_AUDIT',roles)
    freeze(ROOT/f'{stage}_MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells))
    write(PUB/stage/'TRAIN_STATUS.json',dict(status='ALL60_MODEL_CELLS_COMPLETE',stage=stage,fresh_fits=sum(not r['reused'] for r in cells),reused=sum(r['reused'] for r in cells),epochs=len(histories)))
    print('TPRC_ALL60_'+stage.upper()+'_SELECTIONS_COMPLETE',flush=True)

if __name__=='__main__':main(sys.argv[1])
