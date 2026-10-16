"""Exhaustive six-group selection, then refit and seal before HELD labels."""
import argparse
from study_io import *
import ridge


def utility(rows):return float(np.mean([np.nanmean([r[m] for r in rows]) for m in UTILITY]))


def labeled_target(view):
    y=np.full(len(view['patient']),np.nan)
    for role in ['train','cal','choice']:y[view['roles'][role]]=view['role_labels'][role]
    if 'held' in view['roles']:assert np.isnan(y[view['roles']['held']]).all()
    return y


def choice_rows(view,prepared,base,groups,p,y,weights):
    role=view['roles'];cal=role['cal'];choice=role['choice'];pid=view['patient']
    model=ridge.fit(prepared,base,groups,2*y-1,weights,role['train'])
    probability=model['probability'];theta=p.threshold(y[cal].astype(int),probability[cal],pid[cal])
    rows=p.metric_rows(y[choice].astype(int),probability[choice],pid[choice],view['channel'][choice],view['center'][choice],theta)
    score=utility(rows);source=[utility([r for r in rows if r['center']==c]) for c in sorted(set(view['center'][choice]))]
    penalty=float(np.std(source,ddof=1)) if len(source)>1 else 0.
    return dict(k=len(groups),groups=list(groups),choice_utility=score,choice_source_sd=penalty,choice_score=score-.1*penalty,
        threshold=theta,**{m:float(np.nanmean([r[m] for r in rows])) for m in METRICS})


def selection():
    p=primitives();registry=read(ROOT/'INPUT_REGISTRY_PRIVATE.pkl');binding=jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert registry['binding']==binding and binding['source']==source_binding()
    search=[];plans=[];cells=[];roles=[]
    for entry in registry['entries']:
        assert sha(entry['path'])==entry['sha256'];view=read(Path(entry['path']));assert view['binding']==binding
        y=labeled_target(view);tr=view['roles']['train'];pid=view['patient'];f=view['fold'];s=view['screen']
        roles.append(dict(fold=f,screen=s,**{r+'_patients':len(set(pid[ix])) for r,ix in view['roles'].items()},HELD_labels_omitted=True))
        for context in CONTEXTS:
            path=ROOT/'choice_private'/f'fold{f}_screen{s}_{context}.pkl'
            if path.exists():cell=read(path);assert cell['binding']==binding and cell['input_sha256']==entry['sha256']
            else:
                x,base=context_input(view,context);prepared,pre=ridge.prepare(x,tr);weights=p.weights(y[tr].astype(int),pid[tr])
                rows=[choice_rows(view,prepared,base,gg,p,y,weights) for gg in ridge.subsets(view['eligible'])]
                assert np.isfinite([r['choice_score'] for r in rows]).all()
                selected=ridge.choose(rows);other=ridge.alternative(rows,selected);k=selected['k']
                rng=np.random.default_rng(42+1009*f+43*s+100000*k)
                randoms=[sorted(rng.choice(view['eligible'],size=k,replace=False).tolist()) for _ in range(2)]
                arms=dict(BASE=[],SELECTED=selected['groups'],BEST_ALTERNATIVE=other['groups'],RANDOM_A=randoms[0],RANDOM_B=randoms[1],ALL_ELIGIBLE=view['eligible'])
                for kk in range(len(view['eligible'])+1):
                    best=min([r for r in rows if r['k']==kk],key=lambda r:(-r['choice_score'],tuple(r['groups'])))
                    arms[f'K{kk}']=best['groups']
                cell=dict(binding=binding,input_sha256=entry['sha256'],fold=f,screen=s,context=context,pre=pre,
                    rows=rows,selected=selected,arms=arms,eligible=view['eligible'])
                save(path,cell)
            cells.append(dict(fold=f,screen=s,context=context,path=str(path),sha256=sha(path)))
            for row in cell['rows']:search.append(dict(fold=f,screen=s,context=context,**row))
            for arm,groups in cell['arms'].items():plans.append(dict(fold=f,screen=s,context=context,arm=arm,k=len(groups),groups=','.join(map(str,groups)),
                features=';'.join(GROUPS[g] for g in groups),chosen_using_HELD=False,context_base_physiology_groups=7 if context.endswith('OLD7') else 0))
            print('EXACT64_CHOICE_COMPLETE',f,s,context,'fits',len(cell['rows']),'selected_k',cell['selected']['k'],flush=True)
        write(PUB/'SELECTION_PROGRESS.json',dict(status='EXHAUSTIVE_CHOICE_ONLY',complete_context_paths=len(cells),total_context_paths=80))
    frozen=dict(binding=binding,input_registry_sha256=sha(ROOT/'INPUT_REGISTRY_PRIVATE.pkl'),cells=cells)
    freeze(ROOT/'CHOICES_FROZEN_PRIVATE.json',frozen)
    pd.DataFrame(search).to_csv(PUB/'EXHAUSTIVE_CHOICE_RECORD.csv',index=False)
    pd.DataFrame(plans).to_csv(PUB/'FROZEN_SELECTION_PLAN.csv',index=False)
    pd.DataFrame(roles).to_csv(PUB/'INNER_ROLE_ISOLATION.csv',index=False)
    write(PUB/'CHOICE_STATUS.json',dict(status='ALL80_CONTEXT_CHOICES_FROZEN',search_fits=len(search),context_paths=80,
        choice_freeze_sha256=sha(ROOT/'CHOICES_FROZEN_PRIVATE.json'),HELD_labels_used=False,new_Outer_access=False))
    print('ALL80_CONTEXT_CHOICES_FROZEN_BEFORE_ASSESSMENT',flush=True)


def seal():
    p=primitives();binding=jr(ROOT/'INPUT_FREEZE_PRIVATE.json');frozen=jr(ROOT/'CHOICES_FROZEN_PRIVATE.json')
    assert frozen['binding']==binding and source_binding()==binding['source']
    assert jr(PUB/'CHOICE_STATUS.json')['status']=='ALL80_CONTEXT_CHOICES_FROZEN'
    registry=read(ROOT/'INPUT_REGISTRY_PRIVATE.pkl');cells=[];counts=[];actual=0
    for entry in registry['entries']:
        if not entry['screen']:continue
        assert sha(entry['path'])==entry['sha256'];view=read(Path(entry['path']));y=labeled_target(view);role=view['roles']
        refit=np.r_[role['train'],role['choice']];cal=role['cal'];held=role['held'];pid=view['patient'];f=view['fold'];s=view['screen']
        for context in CONTEXTS:
            selected=next(c for c in frozen['cells'] if c['fold']==f and c['screen']==s and c['context']==context)
            assert sha(selected['path'])==selected['sha256'];choice=read(Path(selected['path']))
            target=ROOT/'assessment_private'/f'fold{f}_screen{s}_{context}.pkl'
            if target.exists():result=read(target);assert result['binding']==binding and result['choice_sha256']==selected['sha256']
            else:
                x,base=context_input(view,context);prepared,pre=ridge.prepare(x,refit);weight=p.weights(y[refit].astype(int),pid[refit])
                models={};aliases={}
                for arm,groups in choice['arms'].items():
                    key=','.join(map(str,groups));aliases[arm]=key
                    if key in models:continue
                    model=ridge.fit(prepared,base,groups,2*y-1,weight,refit);prob=model.pop('probability');model.pop('score')
                    threshold=p.threshold(y[cal].astype(int),prob[cal],pid[cal])
                    models[key]=dict(groups=groups,threshold=threshold,model=model,calibration_probability=prob[cal],HELD_probability=prob[held])
                result=dict(binding=binding,choice_sha256=selected['sha256'],input_sha256=entry['sha256'],fold=f,screen=s,context=context,
                    pre=pre,refit=refit,held=held,models=models,aliases=aliases)
                save(target,result)
            actual+=len(result['models']);counts.append(dict(fold=f,screen=s,context=context,logical_arms=len(result['aliases']),actual_refit_models=len(result['models'])))
            cells.append(dict(fold=f,screen=s,context=context,path=str(target),sha256=sha(target)))
            print('ASSESSMENT_SCORES_SEALED',f,s,context,'actual_models',len(result['models']),flush=True)
    freeze(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json',dict(binding=binding,choice_freeze_sha256=sha(ROOT/'CHOICES_FROZEN_PRIVATE.json'),
        cells=cells,no_HELD_labels_in_selector_inputs=True))
    pd.DataFrame(counts).to_csv(PUB/'REFIT_COUNTS.csv',index=False)
    write(PUB/'SEAL_STATUS.json',dict(status='ALL60_CONTEXT_ASSESSMENT_SCORE_CELLS_SEALED',context_assessment_cells=len(cells),actual_refit_models=actual,
        seal_sha256=sha(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json'),new_encoders=0,new_Outer_access=False))
    print('ALL_ASSESSMENT_PROBABILITIES_SEALED_BEFORE_HELD_LABEL_METRICS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','seal']);args=parser.parse_args()
    selection() if args.phase=='select' else seal()
