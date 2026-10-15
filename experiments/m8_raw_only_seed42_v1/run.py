"""Scratch-train only the original M8 raw morphology path, then five frozen Outer folds."""
import gc
import sys
import time
from pathlib import Path

TASK_ROOT=Path(__file__).parent
PARENT_CODE=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
PARENT_RUNTIME=Path(r'D:\m8_raw_backbones_pr_seed42_v1_runtime')
M8_CODE=Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code')
M8_RUNTIME=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime')
OLD_OUTER=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime')
OUT=Path(r'D:\m8_raw_only_seed42_v1_runtime')
sys.path.insert(0,str(PARENT_CODE))
from support import *
from engine import rng,restore
sys.path.insert(0,str(M8_CODE))
P3=load('raw_only_original_m8_training',M8_CODE/'run.py')
RawOnly=load('raw_only_m8_model',TASK_ROOT/'model.py').M8RawOnly
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss


def setup_new():
    lock=jr(TASK_ROOT/'PROTOCOL_LOCK.json')
    oldlock,core,rec,banks,parent=setup(PARENT_RUNTIME)
    source=jr(M8_RUNTIME/'INPUT_FREEZE_PRIVATE.json')
    assert source['code']=={p.name:sha(p) for p in M8_CODE.glob('*.py')}
    assert source['prior_code']=={p.name:sha(p) for p in (M8_CODE.parent.parent/'phase2/code').glob('*.py')}
    assert sha(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['reference_freeze_sha256']
    assert sha(OLD_OUTER/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==lock['reference_seal_sha256']
    assert jr(OLD_OUTER/'public/OUTER_INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    protected={str(p):sha(p) for p in [PARENT_RUNTIME/'INPUT_FREEZE_PRIVATE.json',
        OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json',OLD_OUTER/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',OLD_OUTER/'OUTER_METRICS_PRIVATE.pkl']}
    for r in jr(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selections']:
        protected[r['checkpoint']]=sha(r['checkpoint'])
        assert protected[r['checkpoint']]==r['checkpoint_sha256']
    binding=dict(source={p.name:sha(p) for p in TASK_ROOT.iterdir() if p.suffix in ('.py','.json')},
        parent_binding_sha256=digest(parent),m8_source=source,protected=protected,torch=torch.__version__,numpy=np.__version__)
    OUT.mkdir(exist_ok=True);(OUT/'public').mkdir(exist_ok=True)
    path=OUT/'INPUT_FREEZE_PRIVATE.json'
    if path.exists():assert jr(path)==binding,'Changed source/data/config; do not silently resume'
    else:write(path,binding)
    return lock,core,rec,banks,binding


def build_model(core,rec,fold):
    cp=BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
    # Same constructor and RNG consumption as M8, never a trained raw initializer.
    _,_,fac,_,records,_,_,_,_=P3.context(SOURCE,BASELINE)
    original=P3.make(core,fac,records,cp,42+1009*fold,'M8_MACRO_AWARE')
    oldcp=torch.load(M8_RUNTIME/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert core.state_hash(original.state_dict())==oldcp['binding']['initial']
    model=RawOnly(original).cuda()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in model.state_dict().items())
    assert not any(k.startswith(('base.','gate.')) for k in model.state_dict())
    return model,original


def predict(model,patient,channel,indices,store):
    model.eval();ll=np.empty(len(indices),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(patient[indices])):
            jj=np.flatnonzero(patient[indices]==pid);ii=indices[jj]
            raw,mask,times,_=store.get(str(pid),channel[ii].astype(str).tolist(),'cuda')
            ll[jj]=model(raw,mask,times).cpu().numpy()
    assert np.isfinite(ll).all() and np.max(np.abs(ll))<=1
    return ll


def public_selection(q):
    return {k:v for k,v in q.items() if k!='checkpoint'}


def train(lock,core,rec,banks,binding):
    store=Store(BASELINE);selections=[];histories=[];audits=[]
    for fold,d in enumerate(banks,1):
        seed=42+1009*fold;model,original=build_model(core,rec,fold)
        initial=statehash(model.state_dict());tr=d['train'];va=d['val'];rr=roles(rec['meta'],fold)
        assert set(d['patient'][tr])==rr['fit'] and set(d['patient'][va])==rr['validation']
        assert set(d['patient']).isdisjoint(rr['test'])
        if fold==1:
            pid=sorted(rr['fit'])[0];ii=tr[d['patient'][tr]==pid]
            raw,mask,times,_=store.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
            original.eval();model.eval()
            with torch.inference_mode():
                fake=raw.new_zeros((*mask.shape,7));scale=raw.new_ones(7)
                expected=original.features(raw,mask,times,fake,scale,mode='identity')[0]
                actual=model.features(raw,mask,times);error=float((actual-expected).abs().max())
                assert error==0
                perm=torch.arange(raw.shape[1]-1,-1,-1,device='cuda')
                assert torch.allclose(model.features(raw[:,perm],mask[:,perm],times),actual[perm],rtol=1e-5,atol=1e-5)
                assert torch.equal(model(raw,mask,times),torch.zeros(raw.shape[1],device='cuda'))
            fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda'))
            opt=torch.optim.AdamW(model.parameters(),lr=lock['learning_rate'],weight_decay=lock['weight_decay'])
            y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            for step in range(3):
                opt.zero_grad(set_to_none=True);loss=P3.objective(model(raw,mask,times),y,fn,'M8_MACRO_AWARE');loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert model.head[2].weight.grad.norm()>0
                if step>=1:assert model.cnn[0].weight.grad.norm()>0 and model.head[0].weight.grad.norm()>0
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            write(OUT/'public/SMOKE_AUDIT.json',dict(status='PASS',raw_only_interface=True,trainable_parameters=8433,
                no_base_or_gate_parameters=True,original_M8_identity_gate_feature_max_error=error,
                zero_initial_logit=True,permutation_equivariant=True,three_legal_FIT_gradient_steps_pass=True,smoke_state_discarded=True))
            del model,opt,raw,mask,times,fake,expected,actual,y,loss
            del original
            model,original=build_model(core,rec,fold)
            assert statehash(model.state_dict())==initial
        del original;gc.collect()
        audits.append(dict(fold=fold,initial_state_sha256=initial,exact_original_M8_CNN_head_initialization=True,parameters=8433))
        cell=OUT/f'fold{fold}';cell.mkdir(exist_ok=True)
        bind=dict(source_sha256=digest(binding),fold=fold,initial_state_sha256=initial)
        opt=torch.optim.AdamW(model.parameters(),lr=lock['learning_rate'],weight_decay=lock['weight_decay'])
        core.seed_all(seed)
        fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda'))
        groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
        best=key=None;stale=0;history=[];start=1;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
        if last.exists():
            q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind
            model.load_state_dict(q['state']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
            best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
            if done.exists():
                assert jr(done)['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and jr(done)['last_sha256']==sha(last)
                start=31
            elif q['epoch']>=6 and stale>=6:start=31
        for epoch in range(start,31):
            began=time.perf_counter();model.train();losses=[]
            for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
                ii=groups[int(j)];pid=d['patient'][ii[0]]
                raw,mask,times,_=store.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
                y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
                opt.zero_grad(set_to_none=True);ll=model(raw,mask,times);loss=P3.objective(ll,y,fn,'M8_MACRO_AWARE')
                assert torch.isfinite(loss);loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                del raw,mask,times,y,ll,loss
            ll=predict(model,d['patient'],d['channel'],va,store);p=torch.sigmoid(torch.from_numpy(ll)).numpy()
            sel=core.select_threshold(d['y'][va],p,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
            row=dict(fold=fold,epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-began,
                     threshold=sel['threshold'],validation_macro_f1=sel['macro_f1'],validation_ez_f1=sel['ez_f1'])
            history.append(row)
            if key is None or candidate>key:
                key=candidate;stale=0;best=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    epoch=epoch,threshold=sel['threshold'],validation_macro_f1=sel['macro_f1'],validation_ez_f1=sel['ez_f1'],validation_logit=ll)
            else:stale+=1
            atomic_torch(last,dict(binding=bind,state=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,stale=stale,history=history,epoch=epoch))
            write(OUT/'public/RUN_STATUS.json',dict(status='TRAINING',fold=fold,epoch=epoch,best_epoch=best['epoch'],new_outer_access=False))
            print('M8_RAW_ONLY_EPOCH',fold,epoch,'seconds',round(row['seconds'],2),'best',best['epoch'],flush=True)
            if epoch>=6 and stale>=6:break
        assert best is not None
        if not done.exists():
            atomic_torch(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best))
            write(done,dict(best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last)))
        selections.append(dict(fold=fold,checkpoint=str(cell/'BEST_PRIVATE.pt'),checkpoint_sha256=sha(cell/'BEST_PRIVATE.pt'),
            initial_state_sha256=initial,epoch=best['epoch'],epochs_run=len(history),threshold_nez=best['threshold'],parameters=8433))
        histories+=history
        pd.DataFrame([public_selection(r) for r in selections]).to_csv(OUT/'public/MODEL_SELECTION.csv',index=False)
        pd.DataFrame(histories).to_csv(OUT/'public/TRAINING_HISTORY.csv',index=False)
        print('M8_RAW_ONLY_FOLD_SELECTED',fold,best['epoch'],flush=True)
        del model,opt;gc.collect();torch.cuda.empty_cache()
    write(OUT/'public/INITIALIZATION_AUDIT.json',dict(status='PASS',folds=audits))
    obj=dict(source_sha256=digest(binding),selections=selections,all_five_selected_before_outer=True)
    path=OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if path.exists():assert jr(path)==obj
    else:write(path,obj)
    return obj


def outer(core,rec,freeze):
    meta=rec['meta'];store=Store(BASELINE);cells={};files={};fh=sha(OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for r in freeze['selections']:
        fold=r['fold'];assert sha(r['checkpoint'])==r['checkpoint_sha256']
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        model,original=build_model(core,rec,fold);del original
        model.load_state_dict(cp['state']);model.eval();before=statehash(model.state_dict())
        ix=np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test'])))
        all_logits=np.empty(len(ix),np.float32)
        for pid in sorted(set(meta['patient'][ix])):
            jj=np.flatnonzero(meta['patient'][ix]==pid);gix=ix[jj]
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
            path=OUT/f'fold{fold}/outer_private'/(token+'.pkl')
            if path.exists():q=read(path)
            else:
                with torch.inference_mode():
                    raw,mask,times,_=store.get(str(pid),meta['channel'][gix].astype(str).tolist(),'cuda')
                    ll=model(raw,mask,times).cpu().numpy()
                q=dict(logit_nez=ll,indices=gix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(path,q)
            assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],gix) and 'y' not in q
            all_logits[jj]=q['logit_nez'];files[f'{fold}_{token}']=sha(path)
        assert statehash(model.state_dict())==before
        path=OUT/f'fold{fold}/OUTER_SCORES_PRIVATE.pkl';obj=dict(indices=ix,logit_nez=all_logits,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256'])
        if path.exists():assert np.array_equal(read(path)['logit_nez'],all_logits) and read(path)['freeze_sha256']==fh
        else:save(path,obj)
        cells[fold]=obj;del model;gc.collect();torch.cuda.empty_cache()
        print('M8_RAW_ONLY_OUTER_FOLD_SEALED',fold,len(ix),flush=True)
    seal=OUT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all80_patients_sealed_before_full_labels=True)
    if seal.exists():assert jr(seal)==obj
    else:write(seal,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    frames=[]
    for r in freeze['selections']:
        q=cells[r['fold']];ix=q['indices'];pn=torch.sigmoid(torch.from_numpy(q['logit_nez'])).numpy()
        frames.append(pd.DataFrame(dict(model='M8_RAW_ONLY',version='seed42',fold=r['fold'],patient=meta['patient'][ix],channel=meta['channel'][ix],
            center=meta['center'][ix],y=y[ix],probability=1-pn,prediction=pn<r['threshold_nez'])))
    oldfr=jr(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for r in oldfr['selections']:
        q=read(OLD_OUTER/f"scores_private/{r['arm']}_fold{r['fold']}.pkl")
        assert q['freeze_sha256']==sha(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json') and q['checkpoint_sha256']==r['checkpoint_sha256']
        ix=q['indices'];pn=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        frames.append(pd.DataFrame(dict(model='D1' if r['arm']=='D1' else 'M8_FULL',version='seed42',fold=r['fold'],patient=meta['patient'][ix],channel=meta['channel'][ix],
            center=meta['center'][ix],y=y[ix],probability=1-pn,prediction=pn<r['threshold'])))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame))
    save(OUT/'PREDICTIONS_PRIVATE.pkl',frame);save(OUT/'PATIENT_METRICS_PRIVATE.pkl',patients)
    return frame,patients


def report_and_verify(lock,core,rec,banks,binding,freeze,frame,patients):
    pub=OUT/'public';metrics=EV.CLASSIFICATION+EV.RANKING
    means=lambda g:{m:EV.nanmean(g[m]) for m in metrics}
    main=[];pool=[];folds=[];centers=[];coverage=[];conf=[];intervals=[]
    for name,g in patients.groupby('model'):
        assert len(g)==g.patient.nunique()==80 and g.n.sum()==7635
        main.append(dict(model=name,unit='patient_macro',patients=80,channels=7635,**means(g)))
        for m in metrics:coverage.append(dict(model=name,metric=m,valid_patients=int(np.isfinite(g[m]).sum()),undefined_patients=int((~np.isfinite(g[m])).sum())))
        for k,target in [('fold',folds),('center',centers)]:
            for v,q in g.groupby(k):target.append(dict(model=name,**{k:v},patients=len(q),channels=int(q.n.sum()),**means(q)))
    for name,g in frame.groupby('model'):
        v=EV.classification(g.y,g.probability,g.prediction);pool.append(dict(model=name,unit='pooled_channels',**v));conf.append(dict(model=name,**{k:v[k] for k in EV.COUNTS}))
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));counts=np.stack([(draws==j).sum(1) for j in range(80)],1)
    for ref in ['D1','M8_FULL']:
        a=patients[patients.model==ref].set_index('patient').sort_index();b=patients[patients.model=='M8_RAW_ONLY'].set_index('patient').sort_index()
        assert a.index.equals(b.index) and np.array_equal(a.fold,b.fold)
        delta=b[metrics].to_numpy()-a[metrics].to_numpy();valid=np.isfinite(delta);den=counts@valid.astype(int);num=counts@np.where(valid,delta,0)
        dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        direct=np.nanmean(delta[draws],axis=1)
        assert np.allclose(np.nanpercentile(dist,[2.5,97.5],axis=0),np.nanpercentile(direct,[2.5,97.5],axis=0),rtol=0,atol=1e-12,equal_nan=True)
        for j,m in enumerate(metrics):intervals.append(dict(comparison='M8_RAW_ONLY-minus-'+ref,metric=m,delta=EV.nanmean(delta[:,j]),ci_low=float(np.nanpercentile(dist[:,j],2.5)),ci_high=float(np.nanpercentile(dist[:,j],97.5)),paired_patients=int(valid[:,j].sum()),draws=10000,seed=42,threshold_reoptimized=False))
    for filename,rows in [('main_results',main),('pooled_metrics',pool),('fold_metrics',folds),('center_metrics',centers),('metric_coverage',coverage),('confusion_matrices',conf),('paired_intervals',intervals)]:
        pd.DataFrame(rows).to_csv(pub/(filename+'.csv'),index=False)
    checked=0
    for (name,fold,pid),g in frame.groupby(['model','fold','patient']):
        r=patients[(patients.model==name)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        y=g.y.to_numpy();p=g.probability.to_numpy(dtype=np.float64);b=g.prediction.to_numpy();ff=f1_score(y,b,average=None,labels=[0,1],zero_division=0)
        tn,fp,fn,tp=confusion_matrix(y,b,labels=[0,1]).ravel()
        v=dict(accuracy=accuracy_score(y,b),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),macro_f1=ff.mean(),f1=ff[1],nez_f1=ff[0],log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15)),TP=tp,FP=fp,TN=tn,FN=fn)
        assert all(np.isclose(value,r[k],rtol=0,atol=1e-12) for k,value in v.items());checked+=1
    assert checked==240
    old=pd.DataFrame(read(OLD_OUTER/'OUTER_METRICS_PRIVATE.pkl'))
    mapping=dict(auroc='ez_auroc',auprc='ez_auprc',macro_f1='macro_f1',f1='ez_f1',nez_f1='nez_f1',accuracy='accuracy')
    for name,arm in [('D1','D1'),('M8_FULL','M8_MACRO_AWARE')]:
        a=patients[patients.model==name].set_index(['fold','patient']).sort_index();b=old[old.arm==arm].set_index(['fold','patient']).sort_index()
        assert a.index.equals(b.index)
        for new,prior in mapping.items():assert np.allclose(a[new],b[prior],rtol=0,atol=1e-12,equal_nan=True)
    for r in freeze['selections']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);last=torch.load(Path(r['checkpoint']).parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        want=max(last['history'],key=lambda q:(q['validation_macro_f1'],q['validation_ez_f1'],-q['epoch']))
        assert cp['epoch']==r['epoch']==want['epoch'] and cp['threshold']==r['threshold_nez']==want['threshold']
        d=banks[r['fold']-1];sel=core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(cp['validation_logit'])).numpy(),d['patient'][d['val']])
        assert sel['threshold']==r['threshold_nez']
    for p,h in binding['protected'].items():assert sha(p)==h
    for p in pub.glob('*.csv'):
        columns=pd.read_csv(p,nrows=0).columns
        assert not set(columns)&{'patient','patient_id','channel','channel_name','checkpoint'}
        assert not any('balanced' in c.lower() for c in columns)
    audit=dict(status='PASS',new_raw_only_models=5,trainable_parameters=8433,verified_patient_rows=240,
        patients_per_arm=80,channels_per_arm=7635,only_raw_model_interface=True,original_m8_cnn_head_initializations_identical=True,
        original_M8_objective_and_selection=True,all_five_selections_frozen_before_outer=True,
        reference_primary_patient_metrics_exact=True,protected_parent_hashes_unchanged=True,
        independent_sklearn_checks=True,independent_bootstrap_check=True,bootstrap_draws=10000,bootstrap_seed=42,
        public_privacy_gate=True,scientific_status='EXPLORATORY_REPEATED_TEST')
    write(pub/'INDEPENDENT_VERIFICATION.json',audit)
    write(pub/'FINAL_STATUS.json',dict(**{k:v for k,v in audit.items() if k!='status'},status='M8_RAW_ONLY_FIVE_FOLD_OUTER_COMPLETE',
        freeze_sha256=sha(OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),seal_sha256=sha(OUT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    write(pub/'PUBLIC_SHA256.json',{p.name:sha(p) for p in pub.iterdir() if p.is_file() and p.name!='PUBLIC_SHA256.json'})


def main():
    lock,core,rec,banks,binding=setup_new()
    if (OUT/'public/FINAL_STATUS.json').exists():
        assert jr(OUT/'public/FINAL_STATUS.json')['status']=='M8_RAW_ONLY_FIVE_FOLD_OUTER_COMPLETE'
        print('ALREADY_COMPLETE_DO_NOT_REEVALUATE',flush=True);return
    freeze=train(lock,core,rec,banks,binding)
    frame,patients=outer(core,rec,freeze)
    report_and_verify(lock,core,rec,banks,binding,freeze,frame,patients)
    print('M8_RAW_ONLY_FIVE_FOLD_OUTER_COMPLETE',flush=True)


if __name__=='__main__':main()
