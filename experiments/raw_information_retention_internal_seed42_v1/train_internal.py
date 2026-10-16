"""Prospective bounded own-FIT racing; no Outer/Development loader or score reader."""
import argparse
import copy
import gc
import time
from scipy.special import expit
from common import *
from model import Model,ARMS,CANDIDATES,ABLATIONS
from engine import cached,direct,rng,restore


def evaluate(model,d,ix,store):
    model.eval();logit=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid);ii=ix[loc]
            raw,mask,times=store.get(pid,d['channel'][ii]);q,_=cached(model,raw,mask,times)
            logit[loc]=q.cpu().numpy()
    assert np.isfinite(logit).all();return logit


def calibration(d,ix,logit):
    y=1-d['y'][ix];p=expit(-logit.astype(float));theta=metrics.threshold(y,p,d['patient'][ix],'ez')
    rows=metrics.patient_rows(y,p,d['patient'][ix],d['channel'][ix],d['center'][ix],theta['threshold'])
    return dict(threshold=theta['threshold'],**{m:float(np.nanmean([r[m] for r in rows])) for m in METRICS})


def admit():
    binding=setup();gate=jr(PUB/'INPUT_ADMISSION.json');assert gate['input_binding_sha256']==digest(binding)
    banks=legal_fit_banks(labels=True);paths=role_paths(banks);d=banks[0];rr=paths[(1,1)]['roles']
    tr=np.r_[rr['train'],rr['choice']];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    store=RawStore();raw,mask,times=store.get(pid,d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda');records=[]
    shared={}
    for arm in gate['eligible_arms']:
        a=Model(arm,1).cuda();b=copy.deepcopy(a);initial=state_hash(a.state_dict());before=rng();tick=time.perf_counter()
        loss,score,z=direct(a,raw,mask,times,y,pos);loss.backward();after=rng();direct_seconds=time.perf_counter()-tick
        restore(before);tick=time.perf_counter();value,other,zz=cached(b,raw,mask,times,y,pos);cache_seconds=time.perf_counter()-tick;after_cache=rng()
        assert torch.equal(score,other) and torch.equal(z,zz) and abs(float(loss.detach().cpu())-value)==0
        maximum=0.
        for (name,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==nn and (p.grad is None)==(q.grad is None)
            if p.grad is None:continue
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
            error=float((p.grad-q.grad).abs().max().cpu());maximum=max(maximum,error)
            assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),(arm,name,error)
        assert torch.equal(after[0],after_cache[0]) and all(torch.equal(a,b) for a,b in zip(after[1],after_cache[1]))
        cnn=state_hash(a.cnn.state_dict());head=state_hash(a.head.state_dict())
        shared[arm]=(cnn,head)
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in a.parameters()),
            logit_loss_error=0.,full_gradient_max_error=maximum,RNG_exact=True,
            direct_seconds=direct_seconds,cached_seconds=cache_seconds,initial_sha256=initial,
            discarded_formal_epochs=0))
        del a,b,loss,score,z,other,zz;gc.collect();torch.cuda.empty_cache()
    if 'C1_PRE_SHORT' in shared:assert shared['C0_WINDOW_SHORT']==shared['C1_PRE_SHORT']
    if 'C3_PRE_LONG' in shared:assert shared['C2_WINDOW_LONG']==shared['C3_PRE_LONG']==shared['C4_PRE_LONG_CONTEXT']
    for arm in ABLATIONS:assert shared[arm]==shared['C0_WINDOW_SHORT']
    write(PUB/'GRADIENT_ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),
        actual_complete_patient=True,reports=records,paired_initial_shared_weights_exact=True,
        no_manual_features_or_D1=True,formal_epochs_discarded=0,precision='FP32_TF32_OFF'))
    print('RETENTION_FULL_PATIENT_ADMISSION_PASS',len(records),flush=True)


def fit_cell(fold,screen,arm,d,path,binding,store):
    cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;cell.mkdir(parents=True,exist_ok=True)
    roles=path['roles'];tr=np.r_[roles['train'],roles['choice']];cal=roles['cal'];held=roles['held']
    model=Model(arm,fold).cuda();initial=state_hash(model.state_dict())
    bind=dict(input=binding,fold=fold,screen=screen,arm=arm,initial_sha256=initial,role_sha256=path['role_sha256'])
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    seed=42+1009*fold;seed_all(seed);pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda')
    start=1;best=key=None;history=[];stale=0;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json';bp=cell/'BEST_PRIVATE.pt'
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
        best=q['best'];key=q['key'];history=q['history'];stale=q['stale'];start=q['epoch']+1
        if done.exists():
            finish=jr(done);assert finish['binding_sha256']==digest(bind) and finish['best_sha256']==sha(bp) and finish['last_sha256']==sha(last)
            start=31
        elif q['epoch']>=6 and stale>=6:start=31
    groups=[tr[d['patient'][tr]==p] for p in sorted(set(d['patient'][tr]))]
    for epoch in range(start,31):
        tick=time.perf_counter();losses=[];model.train()
        for number,j in enumerate(np.random.default_rng(seed+epoch).permutation(len(groups)),1):
            ii=groups[int(j)];raw,mask,times=store.get(d['patient'][ii[0]],d['channel'][ii])
            y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);opt.zero_grad(set_to_none=True)
            value,_,_=cached(model,raw,mask,times,y,pos)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(value)
            if number%10==0:write(PUB/'RUN_STATUS.json',dict(status='TRAINING',fold=fold,screen=screen,arm=arm,epoch=epoch,patients_complete=number,patients_total=len(groups),new_Outer_access=False))
            del raw,mask,times,y
        logit=evaluate(model,d,cal,store);selected=calibration(d,cal,logit);candidate=(selected['f1'],selected['macro_f1'],-epoch)
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**selected))
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,selection=selected)
        else:stale+=1
        atomic_torch(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        print('RETENTION_EPOCH',fold,screen,arm,epoch,'seconds',round(history[-1]['seconds'],2),'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None
    if not done.exists():
        atomic_torch(bp,dict(binding=bind,**best));write(done,dict(binding_sha256=digest(bind),best_sha256=sha(bp),last_sha256=sha(last)))
    model.load_state_dict(best['model']);cal_logit=evaluate(model,d,cal,store);replayed=calibration(d,cal,cal_logit)
    assert replayed==best['selection']
    hp=cell/'HELD_PREDICTIONS_PRIVATE.pkl';pb=dict(input=binding,checkpoint_sha256=sha(bp),role_sha256=path['role_sha256'])
    if hp.exists():
        result=read(hp);assert result['binding']==pb
    else:
        held_logit=evaluate(model,d,held,store)
        result=dict(binding=pb,archive_indices=d['archive_indices'][held],logit_nez=held_logit,
            threshold=best['selection']['threshold'],cal_logit_nez=cal_logit,cal_archive_indices=d['archive_indices'][cal],
            selected_epoch=best['epoch'],epochs=len(history),parameters=sum(p.numel() for p in model.parameters()))
        save(hp,result)
    error=float(np.max(np.abs(result['cal_logit_nez']-cal_logit)));assert error<=1e-6
    p=expit(-result['logit_nez'].astype(float));rows=metrics.patient_rows(1-d['y'][held],p,d['patient'][held],d['channel'][held],d['center'][held],result['threshold'])
    records=[dict(fold=fold,screen=screen,arm=arm,**row) for row in rows]
    means={m:float(np.nanmean([r[m] for r in records])) for m in METRICS}
    sources=sorted(set(r['center'] for r in records));source_f1=[np.mean([r['f1'] for r in records if r['center']==s]) for s in sources]
    penalized=means['f1']-.1*(float(np.std(source_f1,ddof=1)) if len(source_f1)>1 else 0.)
    summary=dict(fold=fold,screen=screen,arm=arm,patients=len(rows),parameters=result['parameters'],selected_epoch=best['epoch'],epochs=len(history),
        threshold=result['threshold'],cal_replay_max_error=error,penalized_f1=penalized,**means,
        checkpoint_sha256=sha(bp),held_predictions_sha256=sha(hp))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    print('RETENTION_CELL_COMPLETE',fold,screen,arm,flush=True)
    return summary,records,[dict(fold=fold,screen=screen,arm=arm,**r) for r in history]


def run():
    binding=setup();gate=jr(PUB/'GRADIENT_ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    eligible=jr(PUB/'INPUT_ADMISSION.json')['eligible_arms'];banks=legal_fit_banks(labels=True);paths=role_paths(banks);store=RawStore()
    summaries=[];patients=[];history=[];racing=[]
    for fold,d in enumerate(banks,1):
        first=[]
        for arm in eligible:
            summary,rows,h=fit_cell(fold,1,arm,d,paths[(fold,1)],binding,store)
            summaries.append(summary);patients.extend(rows);history.extend(h);first.append(summary)
            pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
            pd.DataFrame(history).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        base=next(r for r in first if r['arm']=='C0_WINDOW_SHORT');candidates=[r for r in first if r['arm'] in CANDIDATES[1:]]
        if not candidates:
            racing.append(dict(fold=fold,advanced=False,reason='no_identifiable_candidate_family'));continue
        best=max(r['penalized_f1'] for r in candidates)
        chosen=min([r for r in candidates if r['penalized_f1']>=best-.005],key=lambda r:(r['parameters'],-r['penalized_f1'],CANDIDATES.index(r['arm'])))
        advance=chosen['f1']>=base['f1']+.02 and chosen['auroc']>=base['auroc']-.01
        decision=dict(fold=fold,chosen=chosen['arm'],advanced=advance,initial_f1_gain=chosen['f1']-base['f1'],initial_auroc_gain=chosen['auroc']-base['auroc'],
            decision_data='own_FIT_screen1_selection_only_not_independent_confirmation')
        freeze(ROOT/'race_private'/f'fold{fold}.json',decision);racing.append(decision)
        if advance:
            for screen in [2,3]:
                for arm in ['C0_WINDOW_SHORT',chosen['arm']]:
                    summary,rows,h=fit_cell(fold,screen,arm,d,paths[(fold,screen)],binding,store)
                    summaries.append(summary);patients.extend(rows);history.extend(h)
        pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
        pd.DataFrame(history).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        write(PUB/'RACING_DECISIONS.json',dict(status='RACING',decisions=racing))
    assert len(summaries)<=65
    save(ROOT/'PATIENT_RESULTS_PRIVATE.pkl',patients)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,
        cells=[{k:r[k] for k in ['fold','screen','arm','checkpoint_sha256','held_predictions_sha256']} for r in summaries],racing=racing))
    write(PUB/'RUN_STATUS.json',dict(status='ALL_REGISTERED_INTERNAL_CELLS_COMPLETE',actual_cells=len(summaries),maximum_cells=65,
        epochs=sum(r['epochs'] for r in summaries),inner_candidate_selection_only=True,no_original_Development_or_Outer_access=True,
        seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),target_F1_not_established_by_this_internal_study=True))
    print('RETENTION_INTERNAL_TRAIN_COMPLETE',len(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['admit','train']);a=p.parse_args()
    admit() if a.action=='admit' else run()
