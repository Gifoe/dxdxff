"""One frozen final pass for all26 fixed procedures; labels only after score seal."""
import time
from train_integrated_v2 import make
import train_old7_m8 as OLD
from integrated_data_v2 import *
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss

ROOT=OUT/'final_Outer';PUB=ROOT/'public';INTEGRATED=OUT/'integrated_M8_v2'
REFERENCE=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime')


def freeze_context():
    lock,c,core,fac,trainer,records,rec,banks,prior=OLD.setup_new()
    assert jr(TASK/'FINAL_EVALUATION_LOCK.json')['scientific_status']=='EXPLORATORY_REPEATED_TEST'
    assert jr(OLD.RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json')['status']=='PASS'
    assert jr(INTEGRATED/'public/MODEL_FREEZE_AUDIT.json')['status']=='ALL45_LOGICAL_INTEGRATED_MODELS_FROZEN_BEFORE_OUTER'
    assert jr(INTEGRATED/'public/INDEPENDENT_TRAINING_VERIFICATION.json')['status']=='PASS'
    fr=jr(INTEGRATED/'MODEL_FREEZE_PRIVATE.json');assert fr['binding']['prior']==prior
    for n,h in fr['binding']['source'].items():assert sha(TASK/n)==h
    old=read(OLD.RUN/'MODEL_SELECTION_PRIVATE.pkl');assert old['binding']==prior
    selections=[]
    for r in old['selections']:
        selections.append(dict(model='LEGACY_'+r['variant'],kind='legacy',fold=r['fold'],keep=r['keep'],bank=r['bank'],bank_sha256=r['bank_sha256'],
            checkpoint=r['M8_checkpoint'],checkpoint_sha256=r['M8_checkpoint_sha256'],base=r['D1_checkpoint'],base_sha256=r['D1_checkpoint_sha256'],threshold=r['M8_threshold']))
    for r in fr['selections']:
        selections.append(dict(model=r['variant'],kind='integrated',fold=r['fold'],bank=r['bank'],bank_sha256=r['bank_sha256'],
            checkpoint=r['M8_checkpoint'],checkpoint_sha256=r['M8_checkpoint_sha256'],base=r['D1_checkpoint'],base_sha256=r['D1_checkpoint_sha256'],
            threshold=r['M8_threshold'],dmap=r['dmap'],gmap=r['gmap'],wrong_pair=r['wrong_pair'],scale=r['scale'],alias_of=r['alias_of']))
    for fold in range(1,6):
        cp=REFERENCE/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt';q=torch.load(cp,map_location='cpu',weights_only=False)
        selections.append(dict(model='HISTORICAL_M8',kind='historical',fold=fold,bank=str(BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt'),
            checkpoint=str(cp),checkpoint_sha256=sha(cp),base=str(BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'),threshold=q['threshold'],keep=[1]*7))
        selections[-1].update(bank_sha256=sha(selections[-1]['bank']),base_sha256=sha(selections[-1]['base']))
    assert len(selections)==130 and len(set(r['model'] for r in selections))==26
    pool=FeaturePool(rec);pool.populate()
    source=['FINAL_EVALUATION_LOCK.json','evaluate_final.py','verify_integrated_training.py']
    binding=dict(source={n:sha(TASK/n) for n in source},old7_freeze=sha(OLD.RUN/'INPUT_FREEZE_PRIVATE.json'),integrated_freeze=sha(INTEGRATED/'MODEL_FREEZE_PRIVATE.json'),
        old_verifier=sha(OLD.RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json'),integrated_verifier=sha(INTEGRATED/'public/INDEPENDENT_TRAINING_VERIFICATION.json'),
        prior=prior,pool_sources=pool.files)
    for r in selections:
        for name in ['checkpoint','bank','base']:assert sha(r[name])==r[name+'_sha256']
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    obj=dict(binding=binding,selections=selections,all130_armfolds_frozen=True,no_new_Outer_labels_read=True)
    target=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if target.exists():assert jr(target)==obj
    else:write(target,obj)
    return c,core,fac,records,rec,banks,pool,obj,sha(target)


def inference(core,fac,records,rec,banks,pool,fr,freeze_sha):
    meta=rec['meta'];cells={};sealed={};timing=[]
    for r in fr['selections']:
        fold=r['fold'];arm=r['model'];roles=banks[fold-1]['roles'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])))
        assert set(meta['patient'][ix])==roles['test'] and roles['test'].isdisjoint(roles['fit']|roles['validation'])
        bp=torch.load(r['bank'],map_location='cpu',weights_only=False)
        assert set(bp['patient'][bp['train']])==roles['fit'] and set(bp['patient'][bp['val']])==roles['validation']
        assert set(bp['patient']).isdisjoint(roles['test'])
        if r['kind']=='integrated':
            model=make(core,fac,records,fold,Path(r['base']));model.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model'])
            rawx=pool.mapped_summary(r['dmap'],r['wrong_pair'])[ix];x=transform_frozen(rawx,meta['patient'][ix],bp['pre'],core)
            store=MappedInputs(pool,r['gmap'],r['wrong_pair']);scale=torch.tensor(r['scale'],device='cuda')
        else:
            if r['kind']=='historical':
                model=OLD.P3.make(core,fac,records,Path(r['base']),42+1009*fold,'M8_MACRO_AWARE')
            else:
                model,discard=OLD.build(core,fac,records,Path(r['base']),fold,r['keep']);del discard
            model.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model'])
            z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pre=bp['pre']
            x=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
            x=x*np.repeat(np.asarray(r['keep'],np.float32),2)[None]
            store=OLD.P3.Inputs(BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);store.arm='M1_SCALED_DGM'
            scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
        model.eval();before=core.state_hash(model.state_dict());logit=np.empty(len(ix),np.float32);base=logit.copy();t=time.perf_counter();reused=0
        for pid in sorted(roles['test']):
            jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
            cb=dict(freeze=freeze_sha,checkpoint=r['checkpoint_sha256'],x=digest_array(x[jj]),indices_sha256=digest_array(ii),
                input_sources=pool.files[pid],kind=r['kind'],gate=r.get('gmap',r.get('keep')),wrong=r.get('wrong_pair',False),scale=digest_array(scale.cpu().numpy()))
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24];key=hashlib.sha256(json.dumps(cb,sort_keys=True).encode()).hexdigest()
            path=ROOT/'patient_scores_private'/key/(token+'.pkl')
            if path.exists():q=read(path);assert q['binding']==cb;reused+=1
            else:
                raw,mask,times,delta,perms=store.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda')
                with torch.inference_mode():out=model(torch.as_tensor(x[jj],device='cuda'),raw,mask,times,delta,scale)
                q=dict(binding=cb,logit=out[0].cpu().numpy(),base=out[2].cpu().numpy(),indices=ii);save(path,q)
                del raw,mask,times,delta,perms,out
            assert np.array_equal(q['indices'],ii) and np.isfinite(q['logit']).all();logit[jj]=q['logit'];base[jj]=q['base']
        assert core.state_hash(model.state_dict())==before
        result=dict(indices=ix,score_nez=torch.sigmoid(torch.from_numpy(logit)).numpy(),logit=logit,base_logit=base,
            threshold=r['threshold'],freeze_sha256=freeze_sha,checkpoint_sha256=r['checkpoint_sha256'])
        path=ROOT/'armfold_scores_private'/f'{arm}_fold{fold}.pkl'
        if path.exists():prior=read(path);assert prior['freeze_sha256']==freeze_sha and np.array_equal(prior['logit'],logit)
        else:save(path,result)
        cells[arm,fold]=result;sealed[arm+'_'+str(fold)]=sha(path)
        timing.append(dict(model=arm,fold=fold,patients=len(roles['test']),channels=len(ix),seconds=time.perf_counter()-t,reused_patient_cells=reused))
        print('DISCOVERY_OUTER_ARM_FOLD_SEALED',arm,fold,len(cells),'of130',flush=True)
        del model,bp,store;pool.raw.cache.clear();gc.collect();torch.cuda.empty_cache()
    seal=dict(freeze_sha256=freeze_sha,files=sealed,all130_scores_sealed_before_full_labels=True)
    target=ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'
    if target.exists():assert jr(target)==seal
    else:write(target,seal)
    pd.DataFrame(timing).to_csv(PUB/'INFERENCE_TIMING.csv',index=False)
    return cells


def report(cells,fr,rec,freeze_sha):
    assert jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')['all130_scores_sealed_before_full_labels']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    meta=rec['meta'];frames=[]
    for (arm,fold),q in cells.items():
        ix=q['indices'];frames.append(pd.DataFrame(dict(model=arm,version='seed42',fold=fold,patient=meta['patient'][ix],channel=meta['channel'][ix],
            center=meta['center'][ix],y=y[ix],probability=1-q['score_nez'].astype(float),prediction=q['score_nez']<q['threshold'])))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame));assert len(patients)==2080
    save(ROOT/'PREDICTIONS_PRIVATE.pkl',frame);save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',patients)
    keys=EV.CLASSIFICATION+EV.RANKING;means=lambda g:{k:EV.nanmean(g[k]) for k in keys}
    main=[];pooled=[];folds=[];centers=[];coverage=[];conf=[];foldsd=[]
    for arm,g in patients.groupby('model'):
        assert len(g)==g.patient.nunique()==80 and int(g.n.sum())==7635
        main.append(dict(model=arm,unit='patient_macro',patients=80,channels=7635,**means(g)))
        for key in keys:coverage.append(dict(model=arm,metric=key,valid_patients=int(np.isfinite(g[key]).sum()),undefined_patients=int((~np.isfinite(g[key])).sum())))
        for key,target in [('fold',folds),('center',centers)]:
            for value,h in g.groupby(key):target.append(dict(model=arm,unit='patient_macro',**{key:value},patients=len(h),channels=int(h.n.sum()),**means(h)))
        for key in keys:
            v=np.asarray([EV.nanmean(h[key]) for _,h in g.groupby('fold')]);vv=v[np.isfinite(v)]
            foldsd.append(dict(model=arm,metric=key,fold_mean=EV.nanmean(v),fold_sd_ddof1=float(vv.std(ddof=1)) if len(vv)>1 else np.nan,valid_folds=len(vv)))
    for arm,g in frame.groupby('model'):
        v=EV.classification(g.y,g.probability,g.prediction);pooled.append(dict(model=arm,unit='pooled_channels',patients=80,channels=len(g),**v))
        conf.append(dict(model=arm,scope='all',**{k:v[k] for k in EV.COUNTS}))
        for key,target in [('fold',folds),('center',centers)]:
            for value,h in g.groupby(key):
                v=EV.classification(h.y,h.probability,h.prediction)
                target.append(dict(model=arm,unit='pooled_channels',**{key:value},patients=h.patient.nunique(),channels=len(h),**v))
                conf.append(dict(model=arm,scope=key,**{key:value},**{k:v[k] for k in EV.COUNTS}))
    for name,rows in [('main_results',main),('pooled_metrics',pooled),('fold_metrics',folds),('center_metrics',centers),
        ('metric_coverage',coverage),('confusion_matrices',conf),('fold_mean_std',foldsd)]:pd.DataFrame(rows).to_csv(PUB/(name+'.csv'),index=False)
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));counts=np.stack([(draws==i).sum(1) for i in range(80)],1);intervals=[]
    pairs=[(arm,'LEGACY_FULL_RETRAIN') for arm in sorted(patients.model.unique()) if arm!='LEGACY_FULL_RETRAIN']
    pairs += [(arm,'SLOT10_OLD7') for arm in sorted(patients.model.unique()) if arm.startswith('SLOT10') and arm!='SLOT10_OLD7']
    for arm,ref in pairs:
        a=patients[patients.model==arm].set_index('patient').sort_index();b=patients[patients.model==ref].set_index('patient').sort_index()
        assert a.index.equals(b.index) and np.array_equal(a.fold,b.fold)
        delta=a[keys].to_numpy()-b[keys].to_numpy();valid=np.isfinite(delta);den=counts@valid.astype(int)
        dist=np.divide(counts@np.where(valid,delta,0),den,out=np.full((10000,len(keys)),np.nan),where=den>0)
        assert np.allclose(np.nanmean(delta[draws[:256]],axis=1),dist[:256],atol=1e-12,rtol=0,equal_nan=True)
        for j,key in enumerate(keys):
            percentiles=[2.5,97.5];adjusted=arm=='SLOT10_ELIGIBLE_ADD' and key in ['auroc','auprc','macro_f1']
            row=dict(model=arm,reference=ref,metric=key,delta=EV.nanmean(delta[:,j]),ci_low=float(np.nanpercentile(dist[:,j],percentiles[0])),
                ci_high=float(np.nanpercentile(dist[:,j],percentiles[1])),draws=10000,paired_valid_patients=int(valid[:,j].sum()),threshold_reoptimized=False)
            if adjusted:row.update(bonferroni_ci_low=float(np.nanpercentile(dist[:,j],100*.05/12)),bonferroni_ci_high=float(np.nanpercentile(dist[:,j],100*(1-.05/12))))
            intervals.append(row)
    pd.DataFrame(intervals).to_csv(PUB/'paired_intervals.csv',index=False)
    for (arm,fold,pid),g in frame.groupby(['model','fold','patient']):
        r=patients[(patients.model==arm)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        yy=g.y.to_numpy();p=g.probability.to_numpy(dtype=np.float64);pred=g.prediction.to_numpy()
        tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel();assert np.array_equal([tn,fp,fn,tp],[r.TN,r.FP,r.FN,r.TP])
        assert abs(r.nez_f1-f1_score(yy,pred,pos_label=0,zero_division=0))<1e-12
        assert abs(r.macro_f1-f1_score(yy,pred,labels=[0,1],average='macro',zero_division=0))<1e-12
        if yy.sum() and (1-yy).sum():assert abs(r.auroc-roc_auc_score(yy,p))<1e-12
        if yy.sum():assert abs(r.auprc-average_precision_score(yy,p))<1e-12
        assert abs(r.log_loss-log_loss(yy,np.clip(p,1e-15,1-1e-15),labels=[0,1]))<1e-12
        assert abs(r.ez_fraction_bias-(pred.mean()-yy.mean()))<1e-12
        order=g.sort_values(['probability','channel'],ascending=[False,True],kind='stable').y.to_numpy()
        if order.sum():assert abs(r.mrr-1/(np.flatnonzero(order)[0]+1))<1e-12 and r.hit_at_1==order[0]
    meansframe=pd.DataFrame(main).set_index('model');primary=[]
    for ref in ['LEGACY_FULL_RETRAIN','SLOT10_OLD7']:
        delta={k:float(meansframe.loc['SLOT10_ELIGIBLE_ADD',k]-meansframe.loc[ref,k]) for k in ['auroc','auprc','macro_f1','f1']}
        primary.append(dict(reference=ref,**{'delta_'+k:v for k,v in delta.items()},point_estimate_gate=all(delta[k]>=.005 for k in ['auroc','auprc','macro_f1']) and delta['f1']>=0))
    write(PUB/'PRIMARY_SUCCESS_GATE.json',dict(status='PASS_POINT_ESTIMATE_GATE' if all(r['point_estimate_gate'] for r in primary) else 'NO_PREDECLARED_PRIMARY_IMPROVEMENT',
        comparisons=primary,parameters=11690,relative_parameters=11690/11018,not_global_optimality_proof=True,secondary_not_promoted=True))
    for path,h in fr['binding']['prior']['protected'].items():assert sha(path)==h
    for p in PUB.glob('*.csv'):assert not set(pd.read_csv(p,nrows=0).columns)&{'patient','channel','path','checkpoint','probability','prediction','y','indices'}
    write(PUB/'INDEPENDENT_METRIC_VERIFICATION.json',dict(status='PASS',patient_rows=2080,arms=26,all_sklearn_counts_AUC_AP_F1_logloss_ranking_replayed=True,
        bootstrap_256_direct_draws_match=True,bootstrap_draws=10000,freeze_sha256=freeze_sha,seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    write(PUB/'STATUS.json',dict(status='ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED',arms=26,foldcells=130,patient_rows=2080,
        patients_per_arm=80,channels_per_arm=7635,scientific_status='EXPLORATORY_REPEATED_TEST',full_goal_complete=False,
        final_mechanism_compute_report_and_completion_audit_pending=True))


if __name__=='__main__':
    if (PUB/'STATUS.json').exists() and jr(PUB/'STATUS.json')['status']=='ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED':
        print('FINAL_OUTER_ALREADY_COMPLETE_NO_NEW_INFERENCE',flush=True)
    else:
        c,core,fac,records,rec,banks,pool,fr,freeze_sha=freeze_context()
        cells=inference(core,fac,records,rec,banks,pool,fr,freeze_sha);report(cells,fr,rec,freeze_sha)
        print('ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED',flush=True)
