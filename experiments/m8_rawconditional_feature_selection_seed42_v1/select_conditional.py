"""Own-FIT conditional discovery given full192D inner-cross-fitted Raw context."""
import itertools
from conditional_io import *
from conditional_ridge import ConditionalRidge,prepare

METRICS=['accuracy','auroc','auprc','macro_f1','f1','nez_f1','mrr','hit_at_1','ez_fraction_bias']
UTILITY=['auroc','auprc','macro_f1','f1']


def utility(rows):return float(np.mean([np.mean([r[m] for r in rows]) for m in UTILITY]))


def main():
    common,data,training,support,p=old_modules();_,core,_,_,_,_,banks,_=common.legal()
    binding=bindings();assert jr(ROOT/'INPUT_FREEZE_PRIVATE.json')==binding
    assert jr(PUB/'ENCODER_STATUS.json')['status']=='ALL20_INNER_CROSSFIT_ENCODERS_COMPLETE'
    registry=read(ROOT/'ENCODER_REGISTRY_PRIVATE.pkl');pool=read(OLDROOT/'POOL_PRIVATE.pkl');names=pool['names']
    assert registry['binding']==binding
    freeze(ROOT/'SCREEN_INPUT_FREEZE_PRIVATE.json',binding)
    search=[];assessment=[];allpaths=[];final={};bootstrap=[];fit_count=0
    for fold,d in enumerate(banks,1):
        fit=d['train'];pid=d['patient'][fit];channel=d['channel'][fit];center=d['center'][fit];y=(d['y'][fit]==0).astype(int)
        physiology=core.patient_z(pool['summary'][d['archive_indices'][fit]],pid)
        own=[];paths={}
        for screen in range(4):
            ent=next(e for e in registry['entries'] if e['fold']==fold and e['screen']==screen);assert sha(Path(ent['path']))==ent['sha256']
            raw=read(Path(ent['path']));assert np.array_equal(raw['archive_indices'],d['archive_indices'][fit])
            x=np.concatenate([raw['raw'],physiology],1).astype(float);rr=raw['roles'];tr,cal,choice=rr['train'],rr['cal'],rr['choice']
            groups=[]
            for g in range(32):
                values=pool['summary'][d['archive_indices'][fit]][:,2*g:2*g+2]
                nonconstant=np.mean([np.any(np.std(values[tr[pid[tr]==who]],axis=0)>0) for who in np.unique(pid[tr])])
                if np.isfinite(values[tr]).mean()>=.99 and nonconstant>=.5:groups.append(g)
            assert len(groups)>=9
            prepared,pre=prepare(x,tr);weights=p.weights(y[tr],pid[tr]);target=2*y.astype(float)-1
            solver=ConditionalRidge(prepared,target,weights,tr);cache={}

            def assess(model,subset,held):
                value=model.fit(subset);theta=p.threshold(y[cal],value['probability'][cal],pid[cal])
                rows=p.metric_rows(y[held],value['probability'][held],pid[held],channel[held],center[held],theta)
                return value,theta,rows

            def choose(subset,kind):
                nonlocal fit_count
                key=tuple(sorted(subset))
                if key not in cache:
                    _,theta,rows=assess(solver,key,choice);cache[key]=(utility(rows),rows)
                    fit_count+=1
                    search.append(dict(fold=fold,screen=screen,kind=kind,k=len(key),groups=','.join(map(str,key)),features=';'.join(names[g] for g in key),
                        choice_utility=cache[key][0],**{m:float(np.mean([r[m] for r in rows])) for m in METRICS}))
                return cache[key][0]

            path=[dict(k=0,groups=[],utility=choose([],'greedy'))];used=[]
            for k in range(1,10):
                candidates=[(choose(sorted(used+[g]),'greedy'),tuple(-v for v in sorted(used+[g])),sorted(used+[g])) for g in groups if g not in used]
                best=max(candidates,key=lambda r:(r[0],r[1]));used=best[2];path.append(dict(k=k,groups=used.copy(),utility=best[0]))
            paths[screen]=dict(path=path,raw_entry=ent,roles=rr,pre=pre,eligible=groups)
            allpaths += [dict(fold=fold,screen=screen,**row) for row in path]
            if screen:
                refit=np.r_[tr,choice];xx,pp=prepare(x,refit);mm=ConditionalRidge(xx,target,p.weights(y[refit],pid[refit]),refit)
                for recipe,row in [('path',r) for r in path]+[('old7',dict(k=7,groups=list(range(7))))]:
                    _,theta,rows=assess(mm,row['groups'],rr['held']);fit_count+=1
                    own += [dict(fold=fold,screen=screen,recipe=recipe,k=row['k'],**r) for r in rows]
                print('RAW_CONDITIONAL_ASSESSMENT',fold,screen,flush=True)
            else:
                # Keep the final path's sufficient statistics for same-split controls.
                paths[0].update(x=x,prepared=prepared,weights=weights,target=target)
        scores=[]
        for k in range(10):
            rows=[r for r in own if r['recipe']=='path' and r['k']==k];assert len(rows)==len(set(pid))
            bysource=[utility([r for r in rows if r['center']==s]) for s in sorted(set(center))]
            score=utility(rows)-.1*np.std(bysource,ddof=1);scores.append(dict(k=k,score=float(score),utility=utility(rows)))
        best=max(r['score'] for r in scores);k=min(r['k'] for r in scores if r['score']>=best-.005)
        screen=0
        fp=paths[0];selected=fp['path'][k]['groups'];rr=fp['roles'];cal=rr['cal'];choice=rr['choice'];tr=rr['train']
        solver=ConditionalRidge(fp['prepared'],fp['target'],fp['weights'],tr);cache={}
        replacement=[]
        if k:
            for remove in selected:
                for add in fp['eligible']:
                    if add not in selected:
                        s=sorted(set(selected)-{remove}|{add});replacement.append((choose(s,'single_swap'),tuple(-g for g in s),s))
            swap=max(replacement,key=lambda r:(r[0],r[1]))[2]
        else:swap=[]
        rng=np.random.default_rng(42+1009*fold+100000*k)
        randoms=[sorted(rng.choice(fp['eligible'],size=k,replace=False).tolist()) for _ in range(2)]
        final[str(fold)]=dict(k=k,SELECTED=selected,BEST_SWAP=swap,RANDOM_A=randoms[0],RANDOM_B=randoms[1],OLD7=list(range(7)),RAW_ONLY=[],cardinality_scores=scores)
        # Patient-weight perturbation, not full encoder/preprocessing bootstrap.
        train_ids=np.unique(pid[tr]);rng=np.random.default_rng(42+1009*fold+830003)
        for repeat in range(10):
            sample=rng.choice(train_ids,size=len(train_ids),replace=True);multiplicity={who:int((sample==who).sum()) for who in train_ids}
            w=fp['weights']*np.array([multiplicity[who] for who in pid[tr]])
            solver=ConditionalRidge(fp['prepared'],fp['target'],w,tr);cache={};used=[]
            for step in range(1,k+1):
                candidates=[(choose(sorted(used+[g]),f'weight_bootstrap_{repeat}'),tuple(-v for v in sorted(used+[g])),sorted(used+[g])) for g in fp['eligible'] if g not in used]
                used=max(candidates,key=lambda r:(r[0],r[1]))[2]
            bootstrap.append(dict(fold=fold,replicate=repeat,k=k,groups=','.join(map(str,used)),features=';'.join(names[g] for g in used),
                physiology_selected=bool(k),encoder_and_preprocessing_fixed=True,assessment_not_used=True))
        assessment+=own
        save(ROOT/f'fold{fold}_SCREEN_PRIVATE.pkl',dict(binding=binding,paths=paths,final=final[str(fold)]))
        write(PUB/'SCREEN_STATUS.json',dict(status='SCREENING_OWN_FIT_ONLY',completed_folds=fold,total_folds=5))
        print('RAW_CONDITIONAL_FINAL_FOLD',fold,'k',k,'of5',flush=True)
    save(ROOT/'ASSESSMENT_PATIENT_METRICS_PRIVATE.pkl',pd.DataFrame(assessment))
    freeze(ROOT/'SELECTED_FEATURES_PRIVATE.json',dict(binding=binding,folds=final))
    pd.DataFrame(search).to_csv(PUB/'SEARCH_RECORD.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(PUB/'WEIGHT_BOOTSTRAP_RESELECTION.csv',index=False)
    allocated=[]
    for fold,f in final.items():
        for recipe in ['SELECTED','BEST_SWAP','RANDOM_A','RANDOM_B','OLD7','RAW_ONLY']:
            allocated.append(dict(fold=int(fold),recipe=recipe,k=len(f[recipe]),features=';'.join(names[g] for g in f[recipe]),original_validation_and_Outer_used=False))
    pd.DataFrame(allocated).to_csv(PUB/'FROZEN_INTERNAL_SELECTION.csv',index=False)
    pt=pd.DataFrame(assessment);metrics=[];sources=[]
    for (fold,recipe,k),g in pt.groupby(['fold','recipe','k']):
        metrics.append(dict(fold=fold,recipe=recipe,k=k,patient_appearances=len(g),**{m:float(np.mean(g[m])) for m in METRICS}))
        for source,h in g.groupby('center'):sources.append(dict(fold=fold,recipe=recipe,k=k,source=source,patients=len(h),**{m:float(np.mean(h[m])) for m in METRICS}))
    pd.DataFrame(metrics).to_csv(PUB/'ASSESSMENT_CARDINALITY.csv',index=False)
    pd.DataFrame(sources).to_csv(PUB/'ASSESSMENT_SOURCE.csv',index=False)
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal_protection']
    write(PUB/'SCREEN_STATUS.json',dict(status='COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN',actual_ridge_fits=fit_count,assessment_rows=len(pt),
        per_fold_k={f:r['k'] for f,r in final.items()},original_validation_and_Outer_used=False,
        new_neural_candidate_or_Outer_inference=False,positive_minimality_or_increment_not_assumed=True))


if __name__=='__main__':main()
