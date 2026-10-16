"""Independent full low-cost search replay, no validation/Outer feature selection."""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from probes import weights,threshold,metric_rows


def main():
    _,core,_,_,_,rec,banks,deps=legal();pool=read(ROOT/'POOL_PRIVATE.pkl');selected=jr(ROOT/'SELECTED_FEATURES_PRIVATE.json')
    binding=selected['binding']
    for name,h in binding['source'].items():assert sha(HERE/name)==h
    search=pd.read_csv(PUB/'FEATURE_SEARCH_RECORD.csv');checked=0;assessed=0;maximum=0.
    for fold,d in enumerate(banks,1):
        raw=pool['summary'][d['archive_indices']];z=core.patient_z(raw,d['patient']);pid=d['patient'];y=(d['y']==0).astype(int)
        for screen in range(4):
            q=read(ROOT/'screens_private'/(f'fold{fold}_final_path.pkl' if screen==0 else f'fold{fold}_screen{screen}.pkl'))
            assert q['binding']==binding
            roles=[set(pid[q[k]]) for k in ['train','cal','choice']]
            if screen:roles.append(set(pid[q['held']]))
            assert all(roles[a].isdisjoint(roles[b]) for a in range(len(roles)) for b in range(a))
            assert set.union(*roles).issubset(d['roles']['fit']) and set.union(*roles).isdisjoint(d['roles']['validation']|d['roles']['test'])
            good=np.isfinite(z[q['train']]);den=good.sum(0);mu=np.divide(np.nansum(z[q['train']],0),den,out=np.zeros(64),where=den>0)
            fill=np.where(np.isfinite(z),z,mu);ss=StandardScaler().fit(fill[q['train']]);x=ss.transform(fill).astype(np.float32)
            for key,value in [('imputer_statistics',mu),('mean',ss.mean_),('scale',ss.scale_),('var',ss.var_)]:assert np.array_equal(value,q['pre'][key])
            own=search[(search.outer_fold==fold)&(search.screen==screen)];evaluations={}
            for _,row in own.iterrows():
                groups=[int(s) for s in str(row.groups).split(',')];cols=[2*g+j for g in groups for j in [0,1]]
                m=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-6,random_state=42)
                m.fit(x[q['train']][:,cols],y[q['train']],sample_weight=weights(y[q['train']],pid[q['train']]))
                p=m.predict_proba(x[q['cal']][:,cols])[:,1];t=threshold(y[q['cal']],p,pid[q['cal']]);h=q['choice'];p=m.predict_proba(x[h][:,cols])[:,1]
                rr=metric_rows(y[h],p,pid[h],d['channel'][h],d['center'][h],t)
                score=sum(.25*np.nanmean([r[k] for r in rr]) for k in ['auroc','auprc','macro_f1','f1'])
                error=abs(score-float(row.score));assert error<1e-12;maximum=max(maximum,error);checked+=1
                kind=row.get('search_kind','')
                if kind!='equal_k_single_swap':evaluations.setdefault(int(row.k),[]).append((score,tuple(-g for g in groups),groups))
            for step in q['path']:
                best=max(evaluations[step['k']],key=lambda a:(a[0],a[1]))
                assert best[2]==step['groups'] and abs(best[0]-step['choice_score'])<1e-12
            if screen:
                for step in q['predictions']:
                    fit=step['refit'];good=np.isfinite(z[fit]);mu=np.divide(np.nansum(z[fit],0),good.sum(0),out=np.zeros(64),where=good.sum(0)>0)
                    filled=np.where(np.isfinite(z),z,mu);scaler=StandardScaler().fit(filled[fit]);xx=scaler.transform(filled).astype(np.float32)
                    cols=[2*g+j for g in sorted(step['groups']) for j in [0,1]];model=step['prediction']['model'];held=step['held']
                    assert np.array_equal(model.predict_proba(xx[held][:,cols])[:,1],step['prediction']['probability'])
                    assert np.array_equal(model.predict_proba(xx[q['cal']][:,cols])[:,1],step['prediction']['calibration_probability']);assessed+=1
            print('INDEPENDENT_COMPLETE_SEARCH_REPLAY',fold,screen,checked,flush=True)
        final=selected['folds'][str(fold)];scores=final['cardinality_scores'];best=max(r['score'] for r in scores)
        k=min(r['k'] for r in scores if r['score']>=best-.005);assert k==final['k']
        q=read(ROOT/'screens_private'/f'fold{fold}_final_path.pkl');assert q['path'][k-1]['groups']==final['SELECTED']
        swaps=search[(search.outer_fold==fold)&(search.screen==0)&(search.search_kind=='equal_k_single_swap')]
        options=[(float(r.score),tuple(-int(s) for s in str(r.groups).split(',')),[int(s) for s in str(r.groups).split(',')]) for _,r in swaps.iterrows()]
        assert max(options,key=lambda a:(a[0],a[1]))[2]==final['BEST_SWAP']
        rng=np.random.default_rng(42+1009*fold+100000*k)
        for arm in ['RANDOM_A','RANDOM_B']:assert sorted(rng.choice(final['eligible_groups'],size=k,replace=False).tolist())==final[arm]
    assert checked==len(search)==5253 and assessed==135
    write(PUB/'INDEPENDENT_SELECTION_VERIFICATION.json',dict(status='PASS',all_search_fits_independently_retrained=checked,assessment_probability_replays=assessed,
        maximum_choice_score_error=maximum,all_final_cardinalities_features_swaps_randoms_reconstructed=True,roles_preprocessing_and_thresholds_checked=True,
        no_Development_or_Outer_used=True,source_sha256=sha(HERE/'verify_selection.py'),feature_freeze_sha256=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json')))
    print('INDEPENDENT_ALL5253_SEARCH_AND135_ASSESSMENTS_PASS',flush=True)


if __name__=='__main__':main()
