"""Frozen Original backbone + patient-relative linear logit residual; no backbone training."""
import gc
import sys
import time
from pathlib import Path

TASK_ROOT = Path(__file__).parent
PARENT_CODE = Path(r'D:\m8_raw_backbones_pr_seed42_v1')
PARENT_RUNTIME = Path(r'D:\m8_raw_backbones_pr_seed42_v1_runtime')
OUT = Path(r'D:\m8_raw_frozen_pr_linear_residual_seed42_v1_runtime')
sys.path.insert(0, str(PARENT_CODE))
from support import *
from engine import forward
from patient_relative import patient_relative
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, average_precision_score, log_loss, brier_score_loss)


def cache_path(model, fold, role, pid):
    token = hashlib.sha256(str(pid).encode()).hexdigest()[:24]
    return OUT/model/f'fold{fold}'/'representations_private'/role/(token+'.pkl')


def freeze_new(lock, parent_binding):
    files = {p.name: sha(p) for p in sorted(TASK_ROOT.iterdir()) if p.suffix in ('.py', '.json')}
    selections = {}
    protected = {str(PARENT_RUNTIME/'INPUT_FREEZE_PRIVATE.json'): sha(PARENT_RUNTIME/'INPUT_FREEZE_PRIVATE.json')}
    assert protected[str(PARENT_RUNTIME/'INPUT_FREEZE_PRIVATE.json')] == lock['parent_input_sha256']
    for name in lock['models']:
        fr = jr(PARENT_RUNTIME/name/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        assert fr['binding'] == parent_binding
        selections[name] = [r for r in fr['selections'] if r['version'] == 'Original']
        assert len(selections[name]) == 5
        for r in selections[name]:
            assert sha(r['checkpoint']) == r['checkpoint_sha256']
            protected[r['checkpoint']] = r['checkpoint_sha256']
        for f in ('FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_PREDICTIONS_SEAL_PRIVATE.json',
                  'PREDICTIONS_PRIVATE.pkl','PATIENT_METRICS_PRIVATE.pkl'):
            protected[str(PARENT_RUNTIME/name/f)] = sha(PARENT_RUNTIME/name/f)
    binding = dict(source=files, parent_input_sha256=lock['parent_input_sha256'],
                   parent_binding_sha256=digest(parent_binding), selections=selections, protected=protected)
    path = OUT/'INPUT_FREEZE_PRIVATE.json'
    if path.exists(): assert jr(path) == binding, 'Do not resume changed source, protocol or parent'
    else: write(path, binding)
    return binding


def smoke():
    torch.manual_seed(42)
    h = torch.randn(7, 12)
    h[:, 0] = 3
    mask = torch.tensor([1,1,1,1,1,0,0], dtype=torch.bool)
    z = patient_relative(h, mask)
    assert torch.equal(z[~mask], torch.zeros_like(z[~mask])) and torch.equal(z[:,0], torch.zeros(7))
    assert torch.allclose(z[mask].double().mean(0), torch.zeros(12,dtype=torch.float64), atol=1e-7)
    perm = torch.tensor([3,1,5,0,6,4,2])
    assert torch.equal(patient_relative(h[perm], mask[perm]), z[perm])
    net = torch.nn.Linear(12,1); torch.nn.init.zeros_(net.weight); torch.nn.init.zeros_(net.bias)
    ll = torch.randn(7)
    p = 1-torch.sigmoid(ll)
    assert torch.equal(1-torch.sigmoid(ll-net(z).flatten()),p)
    torch.nn.functional.binary_cross_entropy_with_logits(ll-net(z).flatten(), torch.arange(7)%2.).backward()
    assert net.weight.grad.abs().sum() > 0
    return dict(status='PASS', zero_residual_exact_identity=True, nonzero_gradient=True,
                patient_PR_mask_permutation_population_SD=True)


def extract(lock, oldlock, core, rec, binding):
    meta = rec['meta']; store = Store(BASELINE); audits=[]
    source = digest(binding)
    for name in lock['models']:
        for sel in binding['selections'][name]:
            fold=sel['fold']; rr=roles(meta,fold)
            cp=torch.load(sel['checkpoint'],map_location='cpu',weights_only=False)
            model=build(name,False,fold,oldlock,core).eval().requires_grad_(False)
            model.load_state_dict(cp['model'],strict=True); before=statehash(model.state_dict())
            assert all(not p.requires_grad for p in model.parameters())
            for role in ('fit','validation','test'):
                maximum=np.zeros(3); count=0; patients=sorted(rr[role])
                for pid in patients:
                    ix=np.flatnonzero(meta['patient']==pid)
                    path=cache_path(name,fold,role,pid)
                    if path.exists(): q=read(path)
                    else:
                        with torch.inference_mode():
                            raw,mask=raw_patient(store,pid,meta['channel'][ix])
                            ll,h=forward(model,raw,mask,oldlock['chunk'][name])
                            p=1-torch.sigmoid(ll).cpu().numpy()
                            q=dict(indices=ix,embedding=h.cpu().numpy(),logit_nez=ll.cpu().numpy(),probability_ez=p,
                                   source_sha256=source,checkpoint_sha256=sel['checkpoint_sha256'],role=role,
                                   replay_error=[0.,0.,0.])
                        if role=='test':
                            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
                            oldpath=PARENT_RUNTIME/name/f'outer_private/Original/fold{fold}'/(token+'.pkl')
                            old=read(oldpath)
                            assert old['checkpoint_sha256']==sel['checkpoint_sha256'] and np.array_equal(old['indices'],ix)
                            err=[float(np.max(np.abs(p-old['probability_ez']))),
                                 float(np.max(np.abs(q['logit_nez']-old['logit_nez']))),
                                 float(np.max(np.abs(q['embedding']-old['channel_embedding'])))]
                            gate=lock['replay_gate']
                            assert all(a<=b for a,b in zip(err,[gate['max_probability_error'],gate['max_logit_error'],gate['max_embedding_error']])), ('replay gate',name,fold,err)
                            # Exact historical anchor after genuine full encoder/classifier replay parity.
                            q.update(embedding=old['channel_embedding'],logit_nez=old['logit_nez'],
                                     probability_ez=old['probability_ez'],replay_error=err,
                                     historical_cache_sha256=sha(oldpath))
                        save(path,q)
                    assert q['source_sha256']==source and q['checkpoint_sha256']==sel['checkpoint_sha256']
                    assert q['role']==role and np.array_equal(q['indices'],ix) and 'y' not in q
                    assert np.isfinite(q['embedding']).all() and np.isfinite(q['logit_nez']).all()
                    count+=len(ix); maximum=np.maximum(maximum,q['replay_error'])
                audits.append(dict(model=name,fold=fold,role=role,patients=len(patients),channels=count,
                                   outer_score_max_error=maximum[0],outer_logit_max_error=maximum[1],outer_embedding_max_error=maximum[2]))
                print('FROZEN_REPRESENTATIONS',name,fold,role,len(patients),count,flush=True)
            assert statehash(model.state_dict())==before==statehash(cp['model'])
            del model,cp;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(audits).to_csv(OUT/'public/REPRESENTATION_REPLAY_AUDIT.csv',index=False)


def data(name,fold,role,rec,d):
    meta=rec['meta']; rr=roles(meta,fold); out=[]
    lookup={int(i):float(y) for i,y in zip(d['archive_indices'],d['y'])}
    for pid in sorted(rr[role]):
        q=read(cache_path(name,fold,role,pid))
        h=torch.from_numpy(q['embedding']).float()
        z=patient_relative(h)
        assert z.shape==h.shape and torch.isfinite(z).all()
        ix=q['indices']; y=np.array([lookup[int(i)] for i in ix],np.float32)
        out.append(dict(z=z,ll=torch.from_numpy(q['logit_nez']).float(),y=torch.from_numpy(y),
                        patient=np.repeat(str(pid),len(ix)),indices=ix,original_probability=q['probability_ez']))
    return dict(z=torch.cat([q['z'] for q in out]),ll=torch.cat([q['ll'] for q in out]),
                y=torch.cat([q['y'] for q in out]),patient=np.concatenate([q['patient'] for q in out]),
                indices=np.concatenate([q['indices'] for q in out]),
                original_probability=np.concatenate([q['original_probability'] for q in out]))


def validate(net,va):
    with torch.no_grad():p=(1-torch.sigmoid(va['ll']-net(va['z']).flatten())).numpy()
    q=EV.threshold(1-va['y'].numpy(),p,va['patient'])
    return q,p


def train(lock,rec,banks,binding):
    selections=[]; histories=[]; source=digest(binding)
    for name in lock['models']:
        oldva=jr(PARENT_RUNTIME/name/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        for fold,d in enumerate(banks,1):
            tr=data(name,fold,'fit',rec,d);va=data(name,fold,'validation',rec,d)
            assert set(tr['patient']).isdisjoint(set(va['patient']))
            assert set(tr['patient']).isdisjoint(roles(rec['meta'],fold)['test'])
            oldsel=next(q for q in oldva['selections'] if q['version']=='Original' and q['fold']==fold)
            frozen_va=read(Path(oldsel['checkpoint']).parent/'VALIDATION_PRIVATE.pkl')
            pos={int(i):float(p) for i,p in zip(frozen_va['indices'],frozen_va['score_ez'])}
            val_replay=max(abs(float(p)-pos[int(i)]) for i,p in zip(va['indices'],va['original_probability']))
            assert val_replay<=lock['replay_gate']['max_probability_error']
            torch.manual_seed(42+1009*fold)
            net=torch.nn.Linear(tr['z'].shape[1],1);torch.nn.init.zeros_(net.weight);torch.nn.init.zeros_(net.bias)
            selection,p=validate(net,va)
            assert np.max(np.abs(p-va['original_probability']))<=1e-6
            zero_identity=True
            best=dict(state={k:v.clone() for k,v in net.state_dict().items()},epoch=0,selection=selection)
            key=(selection['validation_macro_f1'],selection['validation_positive_f1'],0)
            opt=torch.optim.AdamW(net.parameters(),lr=lock['learning_rate'],weight_decay=lock['weight_decay'])
            ids,inv,counts=np.unique(tr['patient'],return_inverse=True,return_counts=True)
            weights=torch.tensor(1./(len(ids)*counts[inv]),dtype=torch.float32)
            pos_weight=(1-tr['y']).sum()/tr['y'].sum()
            history=[];stale=0;start=1;cell=OUT/name/f'fold{fold}';last=cell/'HEAD_LAST_PRIVATE.pt'
            if last.exists():
                q=torch.load(last,weights_only=False);assert q['source_sha256']==source
                net.load_state_dict(q['state']);opt.load_state_dict(q['optimizer']);best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
                if q['epoch']>=lock['min_epochs'] and stale>=lock['patience']:start=lock['max_epochs']+1
            for epoch in range(start,lock['max_epochs']+1):
                opt.zero_grad(set_to_none=True)
                logits=tr['ll']-net(tr['z']).flatten()
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,tr['y'],pos_weight=pos_weight,reduction='none')*weights).sum()
                loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters())
                torch.nn.utils.clip_grad_norm_(net.parameters(),lock['gradient_clip']);opt.step()
                selected,_=validate(net,va);candidate=(selected['validation_macro_f1'],selected['validation_positive_f1'],-epoch)
                history.append(dict(model=name,fold=fold,epoch=epoch,train_loss=float(loss.detach()),**selected))
                if candidate>key:
                    key=candidate;stale=0;best=dict(state={k:v.detach().clone() for k,v in net.state_dict().items()},epoch=epoch,selection=selected)
                else:stale+=1
                atomic_torch(last,dict(source_sha256=source,state=net.state_dict(),optimizer=opt.state_dict(),best=best,key=key,stale=stale,history=history,epoch=epoch))
                if epoch>=lock['min_epochs'] and stale>=lock['patience']:break
            net.load_state_dict(best['state']);sel,_=validate(net,va);assert sel==best['selection']
            bp=cell/'HEAD_BEST_PRIVATE.pt'
            obj=dict(source_sha256=source,**best)
            if not bp.exists():atomic_torch(bp,obj)
            else: assert statehash(torch.load(bp,weights_only=False)['state'])==statehash(best['state'])
            rr=dict(model=name,fold=fold,checkpoint=str(bp),checkpoint_sha256=sha(bp),base_checkpoint_sha256=oldsel['checkpoint_sha256'],
                    epoch=best['epoch'],epochs_run=len(history),threshold=sel['threshold'],parameters=sum(p.numel() for p in net.parameters()),
                    validation_macro_f1=sel['validation_macro_f1'],validation_ez_f1=sel['validation_positive_f1'],
                    zero_initialization_identity=zero_identity,validation_replay_max_error=val_replay)
            selections.append(rr);histories+=history
            print('PR_LINEAR_HEAD_SELECTED',name,fold,'epoch',rr['epoch'],'of',len(history),flush=True)
    assert len(selections)==20
    pd.DataFrame([{k:v for k,v in q.items() if k!='checkpoint'} for q in selections]).to_csv(OUT/'public/VALIDATION_SELECTION.csv',index=False)
    pd.DataFrame(histories).to_csv(OUT/'public/HEAD_TRAINING_HISTORY.csv',index=False)
    freeze=dict(source_sha256=source,selections=selections,all20_frozen_before_residual_outer=True)
    path=OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if path.exists():assert jr(path)==freeze
    else:write(path,freeze)
    return freeze


def outer(lock,rec,freeze):
    frames=[];sealed={};meta=rec['meta'];freeze_hash=sha(OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for name in lock['models']:
        for r in [q for q in freeze['selections'] if q['model']==name]:
            fold=r['fold'];assert sha(r['checkpoint'])==r['checkpoint_sha256']
            cp=torch.load(r['checkpoint'],weights_only=False)
            ixall=[];scores=[]
            for pid in sorted(roles(meta,fold)['test']):
                q=read(cache_path(name,fold,'test',pid));z=patient_relative(torch.from_numpy(q['embedding']).float())
                net=torch.nn.Linear(z.shape[1],1);net.load_state_dict(cp['state']);net.eval().requires_grad_(False)
                with torch.inference_mode():p=(1-torch.sigmoid(torch.from_numpy(q['logit_nez']).float()-net(z).flatten())).numpy()
                scores.append(p);ixall.append(q['indices'])
            path=OUT/name/f'fold{fold}'/'OUTER_SCORES_PRIVATE.pkl'
            obj=dict(indices=np.concatenate(ixall),score=np.concatenate(scores),freeze_sha256=freeze_hash,checkpoint_sha256=r['checkpoint_sha256'])
            if path.exists():
                previous=read(path);assert previous['freeze_sha256']==freeze_hash and np.array_equal(previous['score'],obj['score'])
            else:save(path,obj)
            sealed[f'{name}_{fold}']=sha(path)
    seal=OUT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'
    obj=dict(freeze_sha256=freeze_hash,files=sealed,all320_residual_patients_sealed_before_full_labels=True)
    if seal.exists():assert jr(seal)==obj
    else:write(seal,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    for name in lock['models']:
        original=read(PARENT_RUNTIME/name/'PREDICTIONS_PRIVATE.pkl')
        original=original[original.version=='Original'].copy();assert len(original)==7635
        frames.append(original)
        for r in [q for q in freeze['selections'] if q['model']==name]:
            q=read(OUT/name/f"fold{r['fold']}"/'OUTER_SCORES_PRIVATE.pkl');ix=q['indices'];p=q['score']
            frames.append(pd.DataFrame(dict(model=name,version='PRLinearResidual',fold=r['fold'],patient=meta['patient'][ix],
                channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],probability=p,prediction=p>=r['threshold'])))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame))
    assert len(patients)==640
    save(OUT/'PREDICTIONS_PRIVATE.pkl',frame);save(OUT/'PATIENT_METRICS_PRIVATE.pkl',patients)
    return frame,patients


def aggregate(lock,frame,patients):
    pub=OUT/'public';metrics=EV.CLASSIFICATION+EV.RANKING
    def means(g):return {m:EV.nanmean(g[m]) for m in metrics}
    main=[];pooled=[];folds=[];centers=[];coverage=[];conf=[];intervals=[];deltas=[]
    for (model,version),g in patients.groupby(['model','version']):
        assert len(g)==80 and g.n.sum()==7635
        main.append(dict(model=model,version=version,unit='patient_macro',patients=80,channels=7635,**means(g)))
        for metric in metrics:coverage.append(dict(model=model,version=version,metric=metric,valid_patients=int(np.isfinite(g[metric]).sum()),undefined_patients=int((~np.isfinite(g[metric])).sum())))
        for key,target in [('fold',folds),('center',centers)]:
            for kk,gg in g.groupby(key):target.append(dict(model=model,version=version,unit='patient_macro',**{key:kk},patients=len(gg),channels=int(gg.n.sum()),**means(gg)))
    for (model,version),g in frame.groupby(['model','version']):
        v=EV.classification(g.y,g.probability,g.prediction);pooled.append(dict(model=model,version=version,unit='pooled_channels',**v))
        conf.append(dict(model=model,version=version,**{k:v[k] for k in EV.COUNTS}))
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));counts=np.stack([(draws==i).sum(1) for i in range(80)],axis=1)
    for name in lock['models']:
        a=patients[(patients.model==name)&(patients.version=='Original')].set_index('patient').sort_index()
        b=patients[(patients.model==name)&(patients.version=='PRLinearResidual')].set_index('patient').sort_index()
        assert a.index.equals(b.index) and np.array_equal(a.fold,b.fold)
        delta=b[metrics].to_numpy()-a[metrics].to_numpy();valid=np.isfinite(delta)
        den=counts@valid.astype(int);num=counts@np.where(valid,delta,0)
        dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        deltas.append(dict(model=name,**{f'delta_{m}':EV.nanmean(delta[:,j]) for j,m in enumerate(metrics)}))
        for j,m in enumerate(metrics):intervals.append(dict(model=name,metric=m,delta=EV.nanmean(delta[:,j]),
            ci_low=float(np.nanpercentile(dist[:,j],2.5)),ci_high=float(np.nanpercentile(dist[:,j],97.5)),valid_paired_patients=int(valid[:,j].sum()),draws=10000,seed=42,threshold_reoptimized=False))
    for filename,rows in [('main_results',main),('pooled_metrics',pooled),('fold_metrics',folds),('center_metrics',centers),
                          ('metric_coverage',coverage),('confusion_matrices',conf),('paired_residual_intervals',intervals),('residual_improvement',deltas)]:
        pd.DataFrame(rows).to_csv(pub/(filename+'.csv'),index=False)
    pd.DataFrame(main)[['model','version',*EV.RANKING]].to_csv(pub/'ranking_metrics.csv',index=False)
    fold_summary=[]
    for (model,version),g in pd.DataFrame(folds).groupby(['model','version']):
        for m in metrics:fold_summary.append(dict(model=model,version=version,metric=m,fold_mean=EV.nanmean(g[m]),fold_std_ddof1=float(np.nanstd(g[m],ddof=1))))
    pd.DataFrame(fold_summary).to_csv(pub/'fold_mean_std.csv',index=False)


def verify(lock,frame,patients,binding):
    checked=0;errors=[];pp=patients.set_index(['model','version','fold','patient'])
    for (model,version,fold,pid),g in frame.groupby(['model','version','fold','patient']):
        y=g.y.to_numpy();p=g.probability.to_numpy(dtype=np.float64);b=g.prediction.to_numpy();r=pp.loc[model,version,fold,pid]
        ff=f1_score(y,b,average=None,labels=[0,1],zero_division=0)
        tn,fp,fn,tp=confusion_matrix(y,b,labels=[0,1]).ravel()
        values=dict(accuracy=accuracy_score(y,b),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),
            macro_f1=ff.mean(),f1=ff[1],nez_f1=ff[0],brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15)),TP=tp,TN=tn,FP=fp,FN=fn)
        assert all(np.isclose(v,r[k],rtol=0,atol=1e-12) for k,v in values.items())
        checked+=1
    assert checked==640
    for name in lock['models']:
        old=read(PARENT_RUNTIME/name/'PATIENT_METRICS_PRIVATE.pkl')
        old=old[old.version=='Original'].set_index(['fold','patient']).sort_index()
        new=patients[(patients.model==name)&(patients.version=='Original')].set_index(['fold','patient']).sort_index()
        assert old.index.equals(new.index)
        assert np.array_equal(old[EV.CLASSIFICATION+EV.RANKING].to_numpy(),new[EV.CLASSIFICATION+EV.RANKING].to_numpy(),equal_nan=True)
    for path,h in binding['protected'].items():assert sha(path)==h, 'Frozen parent changed'
    fr=jr(OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for r in fr['selections']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        dcp=torch.load(r['checkpoint'],weights_only=False)
        assert dcp['epoch']==r['epoch'] and dcp['selection']['threshold']==r['threshold']
    boot=pd.read_csv(OUT/'public/paired_residual_intervals.csv')
    metrics=EV.CLASSIFICATION+EV.RANKING
    for name in lock['models']:
        a=patients[(patients.model==name)&(patients.version=='Original')].set_index('patient').sort_index()
        b=patients[(patients.model==name)&(patients.version=='PRLinearResidual')].set_index('patient').sort_index()
        delta=b[metrics].to_numpy()-a[metrics].to_numpy();draws=np.random.default_rng(42).integers(0,80,size=(10000,80))
        dist=np.nanmean(delta[draws],axis=1)
        for j,m in enumerate(metrics):
            q=boot[(boot.model==name)&(boot.metric==m)].iloc[0]
            want=np.array([EV.nanmean(delta[:,j]),*np.nanpercentile(dist[:,j],[2.5,97.5])]);got=np.array([q.delta,q.ci_low,q.ci_high])
            assert np.allclose(want,got,rtol=0,atol=1e-12,equal_nan=True)
            if np.isfinite(want).any():errors.append(float(np.nanmax(np.abs(want-got))))
    for path in (OUT/'public').glob('*.csv'):
        cols=pd.read_csv(path,nrows=0).columns
        assert not set(cols)&{'patient','patient_id','channel','channel_name','checkpoint'}
        assert not any('balanced' in c.lower() for c in cols)
    audit=dict(status='PASS',configurations=8,residual_heads=20,patient_rows=640,patients_per_configuration=80,channels_per_configuration=7635,
        exact_original_metric_reuse=True,parent_encoder_classifier_buffers_and_predictions_unchanged=True,validation_only_selection=True,
        frozen_before_residual_outer=True,independent_sklearn_checks=True,bootstrap_draws=10000,bootstrap_max_error=max(errors),
        public_privacy_gate=True,scientific_status='EXPLORATORY_REPEATED_TEST')
    write(OUT/'public/INDEPENDENT_VERIFICATION.json',audit)
    write(OUT/'public/FINAL_STATUS.json',dict(status='FROZEN_PR_LINEAR_RESIDUAL_ALL_FIVE_OUTER_COMPLETE',**{k:v for k,v in audit.items() if k!='status'},
        freeze_sha256=sha(OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),seal_sha256=sha(OUT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    write(OUT/'public/PUBLIC_SHA256.json',{p.name:sha(p) for p in sorted((OUT/'public').iterdir()) if p.is_file() and p.name!='PUBLIC_SHA256.json'})


def main():
    OUT.mkdir(exist_ok=True);(OUT/'public').mkdir(exist_ok=True)
    lock=jr(TASK_ROOT/'PROTOCOL_LOCK.json');oldlock,core,rec,banks,parent_binding=setup(PARENT_RUNTIME)
    binding=freeze_new(lock,parent_binding)
    write(OUT/'public/SMOKE_AUDIT.json',smoke())
    if (OUT/'public/FINAL_STATUS.json').exists():
        assert jr(OUT/'public/FINAL_STATUS.json')['status']=='FROZEN_PR_LINEAR_RESIDUAL_ALL_FIVE_OUTER_COMPLETE'
        print('ALREADY_COMPLETE_DO_NOT_REEVALUATE',flush=True);return
    extract(lock,oldlock,core,rec,binding)
    freeze=train(lock,rec,banks,binding)
    frame,patients=outer(lock,rec,freeze)
    aggregate(lock,frame,patients);verify(lock,frame,patients,binding)
    print('FROZEN_PR_LINEAR_RESIDUAL_ALL_FIVE_OUTER_COMPLETE',flush=True)


if __name__=='__main__':main()
