"""Independent frozen score/cache/label/metric/bootstrap audit. Never calls a predictor."""
from evaluate_final_v2 import ROOT,PUB,REFERENCE
from integrated_data_v2 import *
import train_old7_m8 as OLD
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss,balanced_accuracy_score,matthews_corrcoef


def equal(a,b,label):
    assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=1e-12,rtol=0,equal_nan=True),label


def independent_classification(y,p,pred):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=np.asarray(pred,bool);n=len(y)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();pos=int(y.sum());neg=n-pos
    div=lambda a,b:float(a/b) if b else np.nan
    f1=f1_score(y,pred,pos_label=1,zero_division=0) if pos else np.nan
    f0=f1_score(y,pred,pos_label=0,zero_division=0) if neg else np.nan
    precision=div(tp,tp+fp);npv=div(tn,tn+fn);recall=div(tp,pos);specificity=div(tn,neg)
    chance=(pos*(tp+fp)+neg*(tn+fn))/n**2;ece=0.
    bins=np.digitize(p,np.arange(1,10)/10,right=False)
    for k in range(10):
        hit=bins==k
        if hit.any():ece+=float(hit.mean())*abs(float(y[hit].mean())-float(p[hit].mean()))
    denominator=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(accuracy=(tp+tn)/n,precision=precision,recall=recall,specificity=specificity,f1=f1,macro_f1=float(np.nanmean([f1,f0])),nez_f1=f0,
        weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),macro_precision=float(np.nanmean([precision,npv])),macro_recall=float(np.nanmean([recall,specificity])),
        balanced_accuracy=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p) if pos and neg else np.nan,
        auprc=average_precision_score(y,p) if pos else np.nan,mcc=matthews_corrcoef(y,pred) if denominator else np.nan,
        kappa=div((tp+tn)/n-chance,1-chance),npv=npv,fpr=div(fp,neg),fnr=div(fn,pos),brier=float(np.mean((p-y)**2)),
        log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]),ece=ece,predicted_ez_fraction=float(pred.mean()),
        true_ez_fraction=pos/n,ez_fraction_bias=float(pred.mean()-y.mean()),TP=tp,TN=tn,FP=fp,FN=fn,n=n,positives=pos,negatives=neg)


def main():
    status=jr(PUB/'STATUS.json');assert status['status']=='ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED'
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    seal=jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert seal['freeze_sha256']==fh and len(seal['files'])==130
    for n,h in fr['binding']['source'].items():assert sha(TASK/n)==h
    for p,h in fr['binding']['prior']['protected'].items():assert sha(p)==h
    assert sha(OLD.RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json')==fr['binding']['old_verifier']
    assert sha(OUT/'integrated_M8_v2/public/INDEPENDENT_TRAINING_VERIFICATION.json')==fr['binding']['integrated_verifier']
    assert (ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json').stat().st_mtime<(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json').stat().st_mtime
    assert (ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json').stat().st_mtime<=(ROOT/'PREDICTIONS_PRIVATE.pkl').stat().st_mtime
    lock,c,core,fac,trainer,records,rec,banks,prior=OLD.setup_new();assert prior==fr['binding']['prior']
    pool=FeaturePool(rec);pool.populate();assert pool.files==fr['binding']['pool_sources']
    meta=rec['meta'];all_indices={};frames=[];patient_cache_files=set()
    for r in fr['selections']:
        arm=r['model'];fold=r['fold'];roles=banks[fold-1]['roles'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])))
        assert roles['test'].isdisjoint(roles['fit']|roles['validation'])
        for key in ['bank','base','checkpoint']:assert sha(r[key])==r[key+'_sha256']
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);assert cp['threshold']==r['threshold']
        bp=torch.load(r['bank'],map_location='cpu',weights_only=False);assert set(bp['pre']['fit_patient_ids'])==roles['fit']
        if r['kind']=='integrated':
            rr=pool.mapped_summary(r['dmap'],r['wrong_pair'])[ix]
            x=transform_frozen(rr,meta['patient'][ix],bp['pre'],core);gate=r['gmap'];wrong=r['wrong_pair'];scale=np.asarray(r['scale'],np.float32)
        else:
            z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pre=bp['pre'];wrong=False;gate=r['keep']
            x=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)*np.repeat(np.asarray(r['keep'],np.float32),2)[None]
            scale=read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')['scale']
        path=ROOT/'armfold_scores_private'/f'{arm}_fold{fold}.pkl'
        assert sha(path)==seal['files'][arm+'_'+str(fold)];q=read(path)
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and 'y' not in q and q['threshold']==r['threshold']
        assert np.array_equal(q['indices'],ix) and np.array_equal(q['score_nez'],torch.sigmoid(torch.from_numpy(q['logit'])).numpy())
        for pid in sorted(roles['test']):
            jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
            # Reconstruct actual transformed input, rather than trusting a stored x hash.
            expected=dict(freeze=fh,checkpoint=r['checkpoint_sha256'],x=digest_array(x[jj]),indices_sha256=digest_array(ii),
                input_sources=pool.files[pid],kind=r['kind'],gate=gate,wrong=wrong,scale=digest_array(scale))
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24];key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
            pp=ROOT/'patient_scores_private'/key/(token+'.pkl');patient_cache_files.add(str(pp))
            pq=read(pp);assert pq['binding']==expected and np.array_equal(pq['indices'],ii)
            assert np.array_equal(pq['logit'],q['logit'][jj]) and np.array_equal(pq['base'],q['base_logit'][jj])
            assert set(meta['channel'][ii])==set(pool.raw.registry['plans'][pid]['channels'])
        all_indices.setdefault(arm,[]).extend(ix.tolist())
        frames.append(pd.DataFrame(dict(model=arm,version='seed42',fold=fold,patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],
            archive_index=ix,probability=1-q['score_nez'].astype(float),prediction=q['score_nez']<r['threshold'])))
        print('FINAL_FROZEN_SCORE_AND_ALL_PATIENT_CACHE_REPLAY_PASS',arm,fold,flush=True)
    assert len(all_indices)==26 and all(sorted(ix)==list(range(7635)) for ix in all_indices.values())
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and int(y.sum())==1743
    frame=pd.concat(frames,ignore_index=True);frame['y']=y[frame.archive_index.to_numpy()]
    original=read(ROOT/'PREDICTIONS_PRIVATE.pkl').sort_values(['model','fold','patient','channel']).reset_index(drop=True)
    compare=frame.sort_values(['model','fold','patient','channel']).reset_index(drop=True)
    for key in ['model','fold','patient','channel','y','probability','prediction']:assert np.array_equal(original[key],compare[key]),key
    patients=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');patients['balanced_accuracy']=patients['macro_recall']
    assert len(patients)==2080
    for (arm,fold,pid),g in frame.groupby(['model','fold','patient']):
        r=patients[(patients.model==arm)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        yy=g.y.to_numpy();p=g.probability.to_numpy(dtype=float);pred=g.prediction.to_numpy();n=len(yy)
        independent=independent_classification(yy,p,pred)
        for key,value in independent.items():equal(r[key],value,'patient '+key)
        tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel();equal([tn,fp,fn,tp],[r.TN,r.FP,r.FN,r.TP],'confusion')
        equal([r.f1,r.nez_f1,r.macro_f1],[f1_score(yy,pred,pos_label=1,zero_division=0),f1_score(yy,pred,pos_label=0,zero_division=0),f1_score(yy,pred,average='macro',zero_division=0)],'F1')
        equal(r.auroc,roc_auc_score(yy,p),'AUROC');equal(r.auprc,average_precision_score(yy,p),'AP')
        equal(r.accuracy,(tp+tn)/n,'accuracy');equal(r.balanced_accuracy,balanced_accuracy_score(yy,pred),'BA')
        equal(r.log_loss,log_loss(yy,np.clip(p,1e-15,1-1e-15),labels=[0,1]),'logloss');equal(r.brier,np.mean((p-yy)**2),'Brier')
        equal(r.ez_fraction_bias,pred.mean()-yy.mean(),'bias')
        if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn):equal(r.mcc,matthews_corrcoef(yy,pred),'MCC')
        else:assert np.isnan(r.mcc)
        ordered=g.sort_values(['probability','channel'],ascending=[False,True],kind='stable').y.to_numpy()
        positions=np.flatnonzero(ordered);equal(r.mrr,1/(positions[0]+1),'MRR');equal([r.hit_at_1,r.hit_at_3,r.hit_at_5],[bool(ordered[:k].any()) for k in [1,3,5]],'hits')
        equal(r.precision_at_3,ordered[:3].sum()/min(3,n),'precision3');equal(r.recall_at_3,ordered[:3].sum()/yy.sum(),'recall3')
        gain=1/np.log2(np.arange(min(5,n))+2);equal(r.ndcg_at_5,(ordered[:len(gain)]*gain).sum()/gain[:min(int(yy.sum()),len(gain))].sum(),'NDCG')
    keys=EV.CLASSIFICATION+['balanced_accuracy']+EV.RANKING
    for filename,grouping in [('main_results.csv',['model']),('fold_metrics.csv',['model','fold']),('center_metrics.csv',['model','center'])]:
        public=pd.read_csv(PUB/filename);public=public[public.unit=='patient_macro']
        for r in public.to_dict('records'):
            g=patients
            for key in grouping:g=g[g[key]==r[key]]
            assert len(g)==r['patients'] and int(g.n.sum())==r['channels']
            for key in keys:equal(EV.nanmean(g[key]),r[key],filename+' '+key)
    for filename,grouping in [('pooled_metrics.csv',['model']),('fold_metrics.csv',['model','fold']),('center_metrics.csv',['model','center'])]:
        public=pd.read_csv(PUB/filename);public=public[public.unit=='pooled_channels']
        for r in public.to_dict('records'):
            g=frame
            for key in grouping:g=g[g[key]==r[key]]
            actual=independent_classification(g.y,g.probability,g.prediction)
            for key,value in actual.items():equal(value,r[key],filename+' pooled '+key)
    for r in pd.read_csv(PUB/'metric_coverage.csv').to_dict('records'):
        g=patients[patients.model==r['model']];valid=int(np.isfinite(g[r['metric']]).sum())
        assert valid==r['valid_patients'] and 80-valid==r['undefined_patients']
    for r in pd.read_csv(PUB/'fold_mean_std.csv').to_dict('records'):
        g=patients[patients.model==r['model']];values=np.asarray([EV.nanmean(h[r['metric']]) for _,h in g.groupby('fold')]);values=values[np.isfinite(values)]
        equal([EV.nanmean(values),values.std(ddof=1) if len(values)>1 else np.nan],[r['fold_mean'],r['fold_sd_ddof1']],'fold SD')
    for r in pd.read_csv(PUB/'confusion_matrices.csv').to_dict('records'):
        g=frame[frame.model==r['model']]
        if r['scope']!='all':g=g[g[r['scope']]==r[r['scope']]]
        actual=independent_classification(g.y,g.probability,g.prediction)
        for key in EV.COUNTS:equal(r[key],actual[key],'group confusion '+key)
    intervals=pd.read_csv(PUB/'paired_intervals.csv');draw=np.random.default_rng(42).integers(0,80,size=(10000,80))
    assert len(intervals.groupby(['model','reference']))==33
    counts=np.stack([(draw==i).sum(1) for i in range(80)],1)
    for (arm,ref),rows in intervals.groupby(['model','reference']):
        a=patients[patients.model==arm].set_index('patient').sort_index();b=patients[patients.model==ref].set_index('patient').sort_index();assert a.index.equals(b.index)
        delta=a[keys].to_numpy()-b[keys].to_numpy();valid=np.isfinite(delta);den=counts@valid.astype(int)
        dist=np.divide(counts@np.where(valid,delta,0),den,out=np.full((10000,len(keys)),np.nan),where=den>0)
        equal(np.nanmean(delta[draw[:256]],axis=1),dist[:256],'direct bootstrap')
        assert len(rows)==len(keys)
        for r in rows.to_dict('records'):
            j=keys.index(r['metric']);equal(r['delta'],EV.nanmean(delta[:,j]),'bootstrap point')
            equal([r['ci_low'],r['ci_high']],np.nanpercentile(dist[:,j],[2.5,97.5]),'95interval')
            if np.isfinite(r.get('bonferroni_ci_low',np.nan)):
                equal([r['bonferroni_ci_low'],r['bonferroni_ci_high']],np.nanpercentile(dist[:,j],[100*.05/12,100*(1-.05/12)]),'adjusted interval')
    # Aliases are exact complete procedures, not multiple independent training runs.
    nf=jr(OUT/'integrated_M8_v2/MODEL_FREEZE_PRIVATE.json');aliases=0
    for r in nf['selections']:
        if r['alias_of'] is not None:
            left=read(ROOT/'armfold_scores_private'/f"{r['variant']}_fold{r['fold']}.pkl")
            right=read(ROOT/'armfold_scores_private'/f"{r['alias_of']}_fold{r['fold']}.pkl")
            assert np.array_equal(left['score_nez'],right['score_nez']) and left['threshold']==right['threshold'];aliases+=1
    assert aliases==9
    for p in PUB.glob('*.csv'):assert not set(pd.read_csv(p,nrows=0).columns)&{'patient','channel','path','label','y','probability','prediction','indices','checkpoint'}
    write(PUB/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',source_sha256=sha(__file__),arms=26,armfolds=130,patient_rows=2080,
        cohort_channels_per_arm=7635,cohort_patients_per_arm=80,all_ownfold_roles_and_complete_channel_sets=True,
        all_checkpoint_source_and_protected_hashes_unchanged=True,all130_score_content_hashes_and_thresholds_verified=True,
        all_patient_cache_inputs_independently_reconstructed=True,unique_private_patient_cache_files=len(patient_cache_files),aliases_exact=9,
        full_labels_only_after_seal_in_evaluator_and_verifier=True,freeze_seal_prediction_timestamp_order_verified=True,
        all2080_metrics_and_ranking_independently_recomputed=True,all_public_patient_fold_center_tables_verified=True,
        all33_paired_bootstrap_comparisons_reproduced=True,bootstrap_draws=10000,adjusted_primary_intervals_verified=True,
        no_training_or_predictor_forward_called=True,privacy_gate=True,freeze_sha256=fh,seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),full_goal_complete=False))
    print('ALL26_ARMS_130_FOLDS_2080_PATIENTS_CACHES_METRICS_AND_BOOTSTRAPS_INDEPENDENT_PASS',flush=True)


if __name__=='__main__':main()
