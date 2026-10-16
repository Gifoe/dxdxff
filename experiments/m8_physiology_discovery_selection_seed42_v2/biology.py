"""Patient-unit physiological associations; legal FIT labels only, not selection."""
import warnings
from scipy.stats import rankdata, wilcoxon
from common import *

PHASES=['pre','early0to8','late8to20','early_minus_pre','late_minus_pre']


def mean(a,axis):
    finite=np.isfinite(a);num=np.nansum(a,axis=axis);den=finite.sum(axis=axis)
    return np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0)


def effect(x,y):
    good=np.isfinite(x);a=x[good];b=y[good];p=int(b.sum());n=len(b)-p
    if not p or not n:return np.nan
    return float(2*(rankdata(a)[b==1].sum()-p*(p+1)/2)/(p*n)-1)


def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);adjust=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(p);out[order]=np.minimum(adjust,1);return out


def interval(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return np.nan,np.nan,np.nan
    draw=np.random.default_rng(42).integers(0,len(a),(10000,len(a)))
    return float(a.mean()),*np.percentile(a[draw].mean(1),[2.5,97.5]).tolist()


def main():
    _,core,_,_,_,rec,banks,deps=legal();lock=jr(HERE/'BIOLOGY_LOCK.json')
    reg=read(ROOT/'POOL_REGISTRY_PRIVATE.pkl');assert sha(ROOT/'POOL_REGISTRY_PRIVATE.pkl')==jr(PUB/'POOL_STATUS.json')['registry_sha256']
    binding=dict(source={n:sha(HERE/n) for n in ['BIOLOGY_LOCK.json','biology.py','common.py','PROTOCOL_LOCK.json']},pool_registry=sha(ROOT/'POOL_REGISTRY_PRIVATE.pkl'),dependencies=deps)
    freeze(ROOT/'BIOLOGY_INPUT_FREEZE_PRIVATE.json',binding)
    rows=[];trajectories=[];grid=np.arange(-28,20,2)
    for fold,d in enumerate(banks,1):
        labels={int(i):int(y==0) for i,y in zip(d['archive_indices'][d['train']],d['y'][d['train']])}
        for pid in sorted(d['roles']['fit']):
            entry=reg['files'][pid];assert sha(entry['path'])==entry['sha256'];q=read(entry['path'])
            assert set(map(int,q['archive_indices'])).issubset(labels)
            y=np.array([labels[int(i)] for i in q['archive_indices']]);phase=mean(q['phase'],0)
            fields=np.concatenate([phase,phase[:,1:]-phase[:,0:1]],1)
            per_run=np.concatenate([q['phase'],q['phase'][:,:,1:]-q['phase'][:,:,0:1]],2)
            for g,name in enumerate(NAMES):
                for k,pname in enumerate(PHASES):
                    value=effect(fields[:,k,g],y)
                    run=np.array([effect(v[:,k,g],y) for v in per_run]);finite=run[np.isfinite(run)];nonzero=finite[finite!=0]
                    agree=float(max((nonzero>0).sum(),(nonzero<0).sum())/len(nonzero)) if len(nonzero)>=2 else np.nan
                    rows.append(dict(outer_fold=fold,patient=pid,center=str(np.unique(d['center'][d['patient']==pid])[0]),feature=name,phase=pname,effect=value,
                        available_seizures=len(finite),zero_seizure_effects=int((finite==0).sum()),repeated_nonzero_seizures=len(nonzero),cross_seizure_sign_agreement=agree))
            if fold==1:
                # Plot bins do not alter native phase masks or model input.
                for t in grid:
                    inside=q['mask'] & (np.abs(q['times'][:,None,:]-t)<1.01)
                    v=q['values'].copy();v[~inside]=np.nan;per_channel=mean(mean(v,2),0)
                    for g,name in enumerate(NAMES):
                        trajectories.append(dict(patient=pid,feature=name,cache_relative_display_sec=int(t),effect=effect(per_channel[:,g],y)))
            del q
        print('FIT_BIOLOGY_COMPLETE',fold,'of5 overlappingFITcohorts',flush=True)
    frame=pd.DataFrame(rows);save(ROOT/'BIOLOGY_PATIENT_EFFECTS_PRIVATE.pkl',frame)
    output=[]
    for (fold,feature,phase),g in frame.groupby(['outer_fold','feature','phase']):
        a=g.effect.to_numpy();a=a[np.isfinite(a)];mu,lo,hi=interval(a)
        p=1. if not len(a) or np.all(a==0) else float(wilcoxon(a,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
        source=[]
        for center,h in g.groupby('center'):
            v=h.effect.to_numpy();v=v[np.isfinite(v)]
            if len(v)>=5:source.append(bool(np.sign(np.mean(v))==np.sign(mu)))
        agree=g.cross_seizure_sign_agreement.to_numpy();agree=agree[np.isfinite(agree)]
        output.append(dict(outer_fold=fold,feature=feature,phase=phase,patients=len(a),mean_effect=mu,ci_low=lo,ci_high=hi,pvalue=p,
            patient_direction_fraction=float((np.sign(a)==np.sign(mu)).mean()) if len(a) else np.nan,
            eligible_source_groups=len(source),same_direction_source_groups=sum(source),repeat_seizure_patients=len(agree),
            median_seizure_sign_agreement=float(np.median(agree)) if len(agree) else np.nan,scope='primary_unique_FIT' if fold==1 else 'overlapping_FIT_sensitivity'))
    table=pd.DataFrame(output);table['BH_q']=np.nan
    for fold,ix in table.groupby('outer_fold').groups.items():table.loc[ix,'BH_q']=bh(table.loc[ix,'pvalue'].to_numpy())
    table['operationally_reproducible']=((table.BH_q<.05)&((table.ci_low>0)|(table.ci_high<0))&(table.patient_direction_fraction>=.6)&(table.same_direction_source_groups>=3)&(table.repeat_seizure_patients>=10)&(table.median_seizure_sign_agreement>=.7))
    table.to_csv(PUB/'PHYSIOLOGICAL_EFFECTS.csv',index=False)
    sources=[]
    for (fold,feature,phase,center),g in frame.groupby(['outer_fold','feature','phase','center']):
        mu,lo,hi=interval(g.effect.to_numpy());sources.append(dict(outer_fold=fold,feature=feature,phase=phase,center=center,finite_patients=int(np.isfinite(g.effect).sum()),mean_effect=mu,ci_low=lo,ci_high=hi))
    pd.DataFrame(sources).to_csv(PUB/'PHYSIOLOGY_SOURCE_EFFECTS.csv',index=False)
    traj=pd.DataFrame(trajectories);save(ROOT/'TRAJECTORY_PATIENT_EFFECTS_PRIVATE.pkl',traj);curves=[]
    for (feature,t),g in traj.groupby(['feature','cache_relative_display_sec']):
        mu,lo,hi=interval(g.effect.to_numpy());curves.append(dict(feature=feature,cache_relative_display_sec=t,finite_patients=int(np.isfinite(g.effect).sum()),mean_effect=mu,ci_low=lo,ci_high=hi))
    pd.DataFrame(curves).to_csv(PUB/'PHYSIOLOGY_TRAJECTORY.csv',index=False)
    write(PUB/'BIOLOGY_STATUS.json',dict(status='COMPLETE_FIT_ONLY_PHYSIOLOGICAL_ASSOCIATIONS',source=binding['source'],primary_patients=len(banks[0]['roles']['fit']),
        source_effect_fields=len(sources),all_feature_phase_effects=len(table),primary_test_family=32*5,patient_is_replicate=True,Outer_and_Development_labels_used=False,
        feature_selection_uses_these_effects=False,independent_onset_or_clinical_EZ_not_proven=True))
    print('FIT_ONLY_BIOLOGY_ALL160_EFFECTS_COMPLETE',flush=True)


if __name__=='__main__':main()
