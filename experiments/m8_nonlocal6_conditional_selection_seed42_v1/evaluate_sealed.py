"""After global score seal: independent patient metrics and bounded inference."""
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
from study_io import *


def independent_rows(y,prob,pid,channel,center,theta):
    rows=[]
    for who in sorted(set(pid)):
        ix=np.flatnonzero(pid==who);a=y[ix];b=prob[ix];pred=b>=theta;positive=int(a.sum());negative=len(a)-positive
        order=np.lexsort((channel[ix].astype(str),-b));first=np.flatnonzero(a[order]);tp=int(((a==1)&pred).sum());tn=int(((a==0)&~pred).sum())
        rows.append(dict(patient=str(who),center=str(center[ix[0]]),n=len(ix),positives=positive,negatives=negative,
            TP=tp,TN=tn,FP=negative-tn,FN=positive-tp,
            auroc=float(roc_auc_score(a,b)) if positive and negative else np.nan,
            auprc=float(average_precision_score(a,b)) if positive else np.nan,
            macro_f1=float(f1_score(a,pred,labels=[0,1],average='macro',zero_division=0)),
            f1=float(f1_score(a,pred,pos_label=1,zero_division=0)) if positive else np.nan,
            nez_f1=float(f1_score(a,pred,pos_label=0,zero_division=0)) if negative else np.nan,
            accuracy=float(accuracy_score(a,pred)),mrr=1/(int(first[0])+1) if len(first) else np.nan,
            hit_at_1=float(a[order[0]]) if positive else np.nan,ez_fraction_bias=float(pred.mean()-a.mean())))
    return rows


def bootstrap(delta,ids):
    unique=np.unique(ids);rng=np.random.default_rng(42);draws=rng.integers(len(unique),size=(10000,len(unique)))
    weights=np.array([np.bincount(row,minlength=len(unique)) for row in draws],float)
    sums=np.array([np.nansum(delta[ids==who],axis=0) for who in unique]);counts=np.array([np.isfinite(delta[ids==who]).sum(0) for who in unique])
    den=weights@counts;dist=np.divide(weights@sums,den,out=np.full(den.shape,np.nan),where=den>0)
    mean=np.nanmean(delta,axis=0);ci=np.nanpercentile(dist,[2.5,97.5],axis=0)
    return mean,ci,dist


def public_summaries(frame):
    for suffix,keys in [('MAIN',['context','arm']),('FOLD',['context','arm','fold']),('SOURCE',['context','arm','center'])]:
        out=[]
        for key,g in frame.groupby(keys):
            if not isinstance(key,tuple):key=(key,)
            out.append(dict(zip(keys,key))|dict(patient_appearances=len(g),unique_patients=g.patient.nunique(),
                **{m:float(g[m].mean()) for m in METRICS},**{m+'_valid_patient_appearances':int(g[m].notna().sum()) for m in METRICS}))
        pd.DataFrame(out).to_csv(PUB/(suffix+'_RESULTS.csv'),index=False)
    out=[]
    for (context,arm),g in frame.groupby(['context','arm']):
        identity=g.groupby('patient')[METRICS].mean()
        out.append(dict(context=context,arm=arm,unique_patients=len(identity),**{m:float(identity[m].mean()) for m in METRICS}))
    pd.DataFrame(out).to_csv(PUB/'IDENTITY_EQUAL_RESULTS.csv',index=False)


def selection_stability():
    plan=pd.read_csv(PUB/'FROZEN_SELECTION_PLAN.csv');selected=plan[plan.arm=='SELECTED'];freq=[];overlap=[];summary=[]
    for context in CONTEXTS:
        for scope,part in [('assessment_paths',selected[(selected.context==context)&(selected.screen>0)]),('final_paths',selected[(selected.context==context)&(selected.screen==0)])]:
            sets=[set() if pd.isna(v) else set(map(int,str(v).split(','))) for v in part.groups]
            for j,name in enumerate(GROUPS):freq.append(dict(context=context,scope=scope,feature=name,selected_paths=sum(j in s for s in sets),paths=len(sets),frequency=sum(j in s for s in sets)/len(sets)))
            own=[]
            for a,b in itertools.combinations(range(len(sets)),2):
                A,B=sets[a],sets[b];jaccard=len(A&B)/len(A|B) if A|B else 1.
                own.append(jaccard);overlap.append(dict(context=context,scope=scope,pair_a=a,pair_b=b,jaccard=jaccard,
                    both_empty=not bool(A|B),overlapping_training_patients=True))
            summary.append(dict(context=context,scope=scope,paths=len(sets),nonempty_paths=sum(bool(s) for s in sets),
                mean_k=float(part.k.mean()),min_k=int(part.k.min()),max_k=int(part.k.max()),mean_pairwise_jaccard=float(np.mean(own))))
    pd.DataFrame(freq).to_csv(PUB/'SELECTION_FREQUENCY.csv',index=False)
    pd.DataFrame(overlap).to_csv(PUB/'SELECTION_OVERLAP_PAIRS.csv',index=False)
    pd.DataFrame(summary).to_csv(PUB/'SELECTION_STABILITY.csv',index=False)
    return pd.DataFrame(summary)


def contrasts(frame):
    views={(c,a):g.sort_values(['fold','screen','patient']) for (c,a),g in frame.groupby(['context','arm'])}
    pairs=[((c,'SELECTED'),(c,b)) for c in CONTEXTS for b in ['BASE','BEST_ALTERNATIVE','RANDOM_A','RANDOM_B']]
    pairs += [(('PHYS','SELECTED'),('PHYS_OLD7','BASE')),(('RAW','SELECTED'),('RAW_OLD7','BASE')),
              (('RAW_OLD7','BASE'),('RAW','BASE'))]
    intervals=[];utility=[];joint=[]
    for left,right in pairs:
        a=views[left];b=views[right];assert list(zip(a.fold,a.screen,a.patient))==list(zip(b.fold,b.screen,b.patient))
        delta=a[METRICS].to_numpy(float)-b[METRICS].to_numpy(float);ids=a.patient.to_numpy();mu,ci,dist=bootstrap(delta,ids)
        label=dict(context=left[0],arm=left[1],reference_context=right[0],reference_arm=right[1])
        for j,m in enumerate(METRICS):intervals.append(dict(**label,metric=m,delta=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],draws=10000,exploratory=True))
        u=delta[:,[METRICS.index(m) for m in UTILITY]].mean(1)[:,None];mean,limits,_=bootstrap(u,ids)
        utility.append(dict(**label,delta=mean[0],ci_low=limits[0,0],ci_high=limits[1,0],draws=10000,exploratory=True))
        columns=[METRICS.index(m) for m in PRIMARY];draw=dist[:,columns];sd=np.nanstd(draw,axis=0,ddof=1)
        dev=np.divide(np.abs(draw-mu[columns]),sd,out=np.zeros_like(draw),where=sd>0)
        critical=float(np.nanpercentile(np.nanmax(dev,axis=1),95))
        for j,m in zip(columns,PRIMARY):
            se=float(np.nanstd(dist[:,j],ddof=1));joint.append(dict(**label,metric=m,delta=mu[j],ci_low=mu[j]-critical*se,ci_high=mu[j]+critical*se,
                critical=critical,draws=10000,family='six_endpoints_within_this_contrast_not_all_contexts_or_prior_studies',exploratory=True))
    pd.DataFrame(intervals).to_csv(PUB/'PAIRED_INTERVALS.csv',index=False);pd.DataFrame(utility).to_csv(PUB/'UTILITY_INTERVALS.csv',index=False)
    pd.DataFrame(joint).to_csv(PUB/'SIX_ENDPOINT_SIMULTANEOUS_SENSITIVITY.csv',index=False)
    return pd.DataFrame(intervals),pd.DataFrame(utility)


def main():
    seal=jr(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json');binding=jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert seal['binding']==binding and source_binding()==binding['source']
    assert sha(ROOT/'CHOICES_FROZEN_PRIVATE.json')==seal['choice_freeze_sha256']
    for cell in seal['cells']:assert sha(cell['path'])==cell['sha256']
    assert len(seal['cells'])==60 and jr(PUB/'SEAL_STATUS.json')['status']=='ALL60_CONTEXT_ASSESSMENT_SCORE_CELLS_SEALED'
    target=ROOT/'PATIENT_METRICS_PRIVATE.pkl';p=primitives()
    if target.exists():frame=read(target);assert frame.attrs['seal']==sha(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json')
    else:
        rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');registry=read(ROOT/'INPUT_REGISTRY_PRIVATE.pkl');records=[]
        for fold in range(1,6):
            # This is the first stage that indexes HELD labels, after all choices/scores are sealed.
            identity=fit_identity_bank(fold,rec)
            for cell in [c for c in seal['cells'] if c['fold']==fold]:
                result=read(Path(cell['path']));entry=next(e for e in registry['entries'] if e['fold']==fold and e['screen']==cell['screen'])
                assert sha(entry['path'])==entry['sha256']==result['input_sha256'];view=read(Path(entry['path']))
                assert np.array_equal(identity['archive_indices'],view['archive_indices']);held=result['held'];y=identity['own_fit_ez'][held]
                for arm,key in result['aliases'].items():
                    model=result['models'][key];prob=model['HELD_probability'];theta=model['threshold']
                    actual=independent_rows(y,prob,view['patient'][held],view['channel'][held],view['center'][held],theta)
                    parent=p.metric_rows(y,prob,view['patient'][held],view['channel'][held],view['center'][held],theta)
                    np.testing.assert_allclose(pd.DataFrame(actual)[METRICS],pd.DataFrame(parent)[METRICS],atol=1e-12,rtol=0,equal_nan=True)
                    records.extend(dict(fold=fold,screen=cell['screen'],context=cell['context'],arm=arm,new_k=len(model['groups']),**r) for r in actual)
            print('SEALED_PATIENT_METRICS_COMPLETE',fold,flush=True)
        frame=pd.DataFrame(records);frame.attrs['seal']=sha(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json');save(target,frame)
    public_summaries(frame);stability=selection_stability();pairs,utility=contrasts(frame)
    main=pd.read_csv(PUB/'MAIN_RESULTS.csv').set_index(['context','arm']);sources=pd.read_csv(PUB/'SOURCE_RESULTS.csv').set_index(['context','arm','center'])
    u=utility[(utility.context=='RAW_OLD7')&(utility.reference_context=='RAW_OLD7')&(utility.reference_arm=='BASE')].iloc[0]
    clinical=all(main.loc[('RAW_OLD7','SELECTED'),m]>main.loc[('RAW_OLD7','BASE'),m] for m in UTILITY)
    source_deltas={c:float(sources.loc[('RAW_OLD7','SELECTED',c),'auroc']-sources.loc[('RAW_OLD7','BASE',c),'auroc']) for c in SOURCES}
    s=stability[(stability.context=='RAW_OLD7')&(stability.scope=='final_paths')].iloc[0]
    checks=dict(utility_interval_lower_positive=bool(u.ci_low>0),all_four_clinical_deltas_positive=bool(clinical),
        all_known_source_AUROC_deltas_at_least_minus002=bool(all(v>=-.02 for v in source_deltas.values())),
        final_pairwise_Jaccard_at_least_point5=bool(s.mean_pairwise_jaccard>=.5),at_least_four_nonempty_final_sets=bool(s.nonempty_paths>=4))
    write(PUB/'ALLOCATION_GATE.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        utility_delta=float(u.delta),utility_ci_low=float(u.ci_low),utility_ci_high=float(u.ci_high),source_AUROC_deltas=source_deltas,
        previous32_group_nonlinear_gate_still_FAIL=True,automatic_full_M8_or_Outer_authorized=False,independent_confirmation=False))
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal']
    write(PUB/'REPORT_STATUS.json',dict(status='COMPLETE_SEALED_INTERNAL_SIX_GROUP_SCREEN',context_paths=80,assessment_score_cells=60,
        actual_search_fits=jr(PUB/'CHOICE_STATUS.json')['search_fits'],actual_assessment_refits=jr(PUB/'SEAL_STATUS.json')['actual_refit_models'],
        logical_patient_metric_rows=len(frame),patient_appearances_per_arm=len(frame[(frame.context=='RAW')&(frame.arm=='SELECTED')]),
        unique_patients=frame.patient.nunique(),independent_patient_metric_engine='sklearn_and_explicit_ranking',all_logical_patient_rows_dual_checked=True,
        private_metrics_sha256=sha(target),score_seal_sha256=sha(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json'),report_source_sha256=sha(HERE/'evaluate_sealed.py'),
        new_encoder_training=0,new_Outer_inference=False,repeated_internal_and_historical_test_exploratory=True,
        allocation_gate='PASS' if all(checks.values()) else 'FAIL',goal_complete_proven=False))
    print('SEALED_INTERNAL_SIX_GROUP_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
