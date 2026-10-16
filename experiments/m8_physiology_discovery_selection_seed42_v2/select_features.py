"""Nested per-outer-FIT cardinality/forward selection with unseen assessments."""
import argparse
import time
from sklearn.model_selection import GroupKFold
from common import *
from probes import fit_probe, preprocess

WEIGHTS = dict(auroc=.25, auprc=.25, macro_f1=.25, f1=.25)


def utility(rows):
    return sum(w*np.nanmean([r[m] for r in rows]) for m,w in WEIGHTS.items())


def columns(groups):
    return [2*g+j for g in sorted(groups) for j in [0,1]]


def split3(remaining, pid, seed):
    ids = np.array(sorted(set(pid[remaining]))); order = np.random.default_rng(seed).permutation(ids)
    n = max(1, int(np.ceil(.2*len(ids))))
    cal = remaining[np.isin(pid[remaining], order[:n])]
    choice = remaining[np.isin(pid[remaining], order[n:2*n])]
    train = remaining[~np.isin(pid[remaining], order[:2*n])]
    sets = [set(pid[q]) for q in [train,cal,choice]]
    assert all(sets[a].isdisjoint(sets[b]) for a in range(3) for b in range(a)) and min(map(len,sets)) > 0
    return train,cal,choice


def eligible(raw, pid, fit, quality):
    eligible=[];audit=[]
    for g in range(32):
        nonconstant=0
        for p in sorted(set(pid[fit])):
            v=raw[fit[pid[fit]==p],2*g:2*g+2]
            nonconstant+=int(np.any(np.nanstd(v,axis=0)>0))
        rate=nonconstant/len(set(pid[fit]));passed=quality[g]['finite_fraction']>=.99 and rate>=.5
        duplicate=None
        if passed:
            for previous in eligible:
                if np.array_equal(raw[fit,2*g:2*g+2],raw[fit,2*previous:2*previous+2],equal_nan=True):
                    duplicate=previous;passed=False;break
        if passed:eligible.append(g)
        audit.append(dict(feature=NAMES[g],group=g,FIT_nonconstant_patient_fraction=rate,window_finite_fraction=quality[g]['finite_fraction'],eligible=passed,
            duplicate_of=None if duplicate is None else NAMES[duplicate],reason='ELIGIBLE' if passed else 'EXACT_FIT_DUPLICATE' if duplicate is not None else 'NUMERIC_AVAILABILITY_GATE_NOT_PHYSIOLOGICAL_INEFFECTIVENESS'))
    assert len(eligible)>=9 and set(range(7)).issubset(eligible)
    return eligible,audit


def path_search(fold,screen,remaining,d,z,groups,seed,binding):
    pid=d['patient'];y=(d['y']==0).astype(int)
    train,cal,choice=split3(remaining,pid,seed)
    x,pre=preprocess(z,train)
    cache={};record=[];path=[];used=[]
    for k in range(1,10):
        candidates=[]
        for g in groups:
            if g in used:continue
            subset=tuple(sorted(used+[g]));key=','.join(map(str,subset))
            q=fit_probe(x,y,pid,d['channel'],d['center'],train,cal,choice,columns(subset),seed)
            score=float(utility(q['rows']));row=dict(outer_fold=fold,screen=screen,k=k,groups=key,features=';'.join(NAMES[j] for j in subset),score=score,
                warnings=';'.join(q['warnings']),iterations=q['iterations'],**{m:float(np.nanmean([r[m] for r in q['rows']])) for m in METRICS})
            record.append(row);candidates.append((score,tuple(-v for v in subset),subset,q))
        best=max(candidates,key=lambda a:(a[0],a[1]));used=list(best[2]);path.append(dict(k=k,groups=used.copy(),choice_score=best[0]))
        print('NESTED_GREEDY_PATH',fold,screen,k,len(record),'candidate fits',flush=True)
    return dict(path=path,search=record,train=train,cal=cal,choice=choice,pre=pre,eligible_groups=groups)


def main():
    c,core,fac,trainer,records,rec,banks,deps=legal();metric=ev()
    status=jr(PUB/'POOL_STATUS.json');assert status['status']=='COMPLETE_UNLABELED32_POOL'
    p=ROOT/'POOL_PRIVATE.pkl';assert sha(p)==status['pool_sha256'];pool=read(p)
    assert pool['names']==NAMES and np.array_equal(pool['summary'][:,:14],rec['R14'])
    quality=pd.read_csv(PUB/'POOL_QUALITY.csv').to_dict('records')
    binding=dict(source={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','common.py','probes.py','select_features.py']},dependencies=deps,
        pool_sha256=sha(p),pool_registry_sha256=status['registry_sha256'],pool_quality_sha256=sha(PUB/'POOL_QUALITY.csv'),seed=42)
    freeze(ROOT/'SCREEN_INPUT_FREEZE_PRIVATE.json',binding)
    all_search=[];assess=[];elig=[];isolation=[];final={};paths=[]
    for fold,d in enumerate(banks,1):
        fit=d['train'];raw=pool['summary'][d['archive_indices']];pid=d['patient'];y=(d['y']==0).astype(int)
        z=core.patient_z(raw,pid);groups,audit=eligible(raw,pid,fit,quality)
        elig += [dict(outer_fold=fold,**r) for r in audit]
        splits=GroupKFold(n_splits=3,shuffle=True,random_state=42+1009*fold)
        own=[]
        for screen,(it,iv) in enumerate(splits.split(z[fit],groups=pid[fit]),1):
            remaining=fit[it];held=fit[iv];target=ROOT/'screens_private'/f'fold{fold}_screen{screen}.pkl'
            inner_groups,inner_audit=eligible(raw,pid,remaining,quality)
            if target.exists():q=read(target);assert q['binding']==binding
            else:
                q=path_search(fold,screen,remaining,d,z,inner_groups,42+1009*fold+43*screen,binding)
                assert set(pid[held]).isdisjoint(set(pid[remaining]))
                refit=np.concatenate([q['train'],q['choice']]);x,pre=preprocess(z,refit);prediction=[]
                for row in q['path']:
                    qq=fit_probe(x,y,pid,d['channel'],d['center'],refit,q['cal'],held,columns(row['groups']),42+1009*fold)
                    # Independent authoritative engine at all nine assessed cardinalities.
                    frame=pd.DataFrame(dict(model='PROBE',version='assessment',fold=fold,patient=pid[held],channel=d['channel'][held],center=d['center'][held],y=y[held],probability=qq['probability'],prediction=qq['probability']>=qq['threshold']))
                    independent=pd.DataFrame(metric.patient_rows(frame)).sort_values('patient')
                    ours=pd.DataFrame(qq['rows']).sort_values('patient')
                    assert np.allclose(independent[METRICS],ours[METRICS],rtol=0,atol=1e-12,equal_nan=True)
                    prediction.append(dict(k=row['k'],groups=row['groups'],prediction=qq,pre=pre,refit=refit,held=held))
                q=dict(q,binding=binding,predictions=prediction,held=held);save(target,q)
            assert q['eligible_groups']==inner_groups
            all_search+=q['search'];paths += [dict(outer_fold=fold,screen=screen,eligible_groups=inner_groups,**r) for r in q['path']]
            elig += [dict(outer_fold=fold,screen=screen,**r) for r in inner_audit]
            sets=[set(pid[q[k]]) for k in ['train','cal','choice','held']]
            assert all(sets[a].isdisjoint(sets[b]) for a in range(4) for b in range(a))
            assert set.union(*sets)==d['roles']['fit'] and set.union(*sets).isdisjoint(d['roles']['validation']|d['roles']['test'])
            isolation.append(dict(outer_fold=fold,screen=screen,selection_train_patients=len(sets[0]),calibration_patients=len(sets[1]),feature_choice_patients=len(sets[2]),assessment_patients=len(sets[3]),original_validation_and_outer_excluded=True,all_four_roles_disjoint=True))
            for r in q['predictions']:
                own += [dict(outer_fold=fold,screen=screen,k=r['k'],**row) for row in r['prediction']['rows']]
        means=[]
        for k in range(1,10):
            a=[r for r in own if r['k']==k];assert len(a)==len(set(pid[fit]))
            percenter=[utility([r for r in a if r['center']==center]) for center in sorted(set(r['center'] for r in a))]
            score=float(utility(a)-.1*np.std(percenter,ddof=1));means.append(dict(k=k,score=score,utility=float(utility(a)),source_utility_sd=float(np.std(percenter,ddof=1))))
        best=max(q['score'] for q in means);k=min(q['k'] for q in means if q['score']>=best-.005)
        target=ROOT/'screens_private'/f'fold{fold}_final_path.pkl'
        if target.exists():q=read(target);assert q['binding']==binding
        else:
            q=dict(path_search(fold,0,fit,d,z,groups,42+1009*fold+170003,binding),binding=binding);save(target,q)
        all_search+=q['search'];paths += [dict(outer_fold=fold,screen=0,eligible_groups=groups,**r) for r in q['path']]
        selected=q['path'][k-1]['groups'];x,_=preprocess(z,q['train']);swaps=[]
        for remove in selected:
            for add in groups:
                if add in selected:continue
                subset=tuple(sorted(set(selected)-{remove}|{add}))
                qq=fit_probe(x,y,pid,d['channel'],d['center'],q['train'],q['cal'],q['choice'],columns(subset),42+1009*fold)
                score=float(utility(qq['rows']));swaps.append((score,tuple(-g for g in subset),list(subset)))
                all_search.append(dict(outer_fold=fold,screen=0,k=k,groups=','.join(map(str,subset)),features=';'.join(NAMES[j] for j in subset),score=score,search_kind='equal_k_single_swap',removed=NAMES[remove],added=NAMES[add]))
        replacement=max(swaps,key=lambda a:(a[0],a[1]))[2]
        # These draws depend only on seed/fold/k and label-blind eligible groups, never scores.
        rng=np.random.default_rng(42+1009*fold+100000*k)
        randoms=[sorted(rng.choice(groups,size=k,replace=False).tolist()) for _ in range(2)]
        final[str(fold)]=dict(k=k,SELECTED=selected,BEST_SWAP=replacement,RANDOM_A=randoms[0],RANDOM_B=randoms[1],OLD7=list(range(7)),
            names={arm:[NAMES[j] for j in gg] for arm,gg in [('SELECTED',selected),('BEST_SWAP',replacement),('RANDOM_A',randoms[0]),('RANDOM_B',randoms[1]),('OLD7',list(range(7)))]},cardinality_scores=means,eligible_groups=groups)
        assess+=own
        write(PUB/'SCREEN_STATUS.json',dict(status='SCREENING',folds_completed=fold,total_folds=5,own_outer_and_development_excluded=True))
        print('FOLD_LOCAL_SELECTION_FROZEN',fold,'k',k,final[str(fold)]['names']['SELECTED'],flush=True)
    save(ROOT/'ASSESSMENT_PATIENTS_PRIVATE.pkl',pd.DataFrame(assess))
    save(ROOT/'SEARCH_PATHS_PRIVATE.pkl',dict(binding=binding,paths=paths))
    freeze(ROOT/'SELECTED_FEATURES_PRIVATE.json',dict(binding=binding,folds=final,selected_inside_each_originalFIT=True,no_Development_or_Outer_selection=True))
    pd.DataFrame(all_search).to_csv(PUB/'FEATURE_SEARCH_RECORD.csv',index=False)
    pd.DataFrame(elig).to_csv(PUB/'FIT_ELIGIBILITY.csv',index=False)
    pd.DataFrame(isolation).to_csv(PUB/'NESTED_ROLE_ISOLATION.csv',index=False)
    public=[]
    for fold,value in final.items():
        for arm in ['SELECTED','BEST_SWAP','RANDOM_A','RANDOM_B','OLD7']:
            public.append(dict(outer_fold=int(fold),arm=arm,k=len(value[arm]),groups=','.join(map(str,value[arm])),features=';'.join(value['names'][arm]),chosen_without_Development_and_Outer=True))
    pd.DataFrame(public).to_csv(PUB/'FROZEN_FEATURE_SELECTION.csv',index=False)
    frame=pd.DataFrame(assess)
    rows=[]
    for (fold,k),g in frame.groupby(['outer_fold','k']):
        rows.append(dict(outer_fold=fold,k=k,patients=len(g),**{m:float(np.nanmean(g[m])) for m in METRICS}))
    pd.DataFrame(rows).to_csv(PUB/'CARDINALITY_ASSESSMENT_BY_FOLD.csv',index=False)
    frequency=[];stability=[]
    for k in range(1,10):
        selected=[r for r in paths if r['k']==k and r['screen']>0]
        for g,name in enumerate(NAMES):frequency.append(dict(k=k,feature=name,selected_paths=sum(g in r['groups'] for r in selected),total_paths=len(selected),frequency=sum(g in r['groups'] for r in selected)/len(selected),paths_overlap=True))
        for i in range(len(selected)):
            for j in range(i):
                a,b=set(selected[i]['groups']),set(selected[j]['groups'])
                e1,e2=set(selected[i]['eligible_groups']),set(selected[j]['eligible_groups'])
                chance=k*k*len(e1&e2)/(len(e1)*len(e2))
                stability.append(dict(k=k,jaccard=len(a&b)/len(a|b),chance_adjusted_overlap=(len(a&b)-chance)/(k-chance),overlapping_training_sets=True))
    pd.DataFrame(frequency).to_csv(PUB/'SELECTION_FREQUENCY.csv',index=False)
    pd.DataFrame(stability).to_csv(PUB/'SELECTION_STABILITY.csv',index=False)
    write(PUB/'SCREEN_STATUS.json',dict(status='ALL5_FIT_ONLY_SELECTIONS_FROZEN',feature_freeze_sha256=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json'),assessment_rows=len(frame),
        actual_probe_fits=len(all_search)+135,nested_assessment_cells=135,selection_paths=20,screened_groups=32,original_validation_and_Outer_used=False,
        source=binding['source'],feature_search_rows=len(all_search),global_consensus_not_used_as_evaluated_subset=True))
    print('ALL5_FIT_ONLY_SELECTIONS_FROZEN',flush=True)


if __name__=='__main__':main()
