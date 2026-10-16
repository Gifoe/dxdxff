"""Replay every dense head and choice; independently audit isolation and inputs."""
import itertools
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from study_io import *


def independent_prepare(x,train):
    valid=np.isfinite(x[train]);counts=valid.sum(0)
    impute=np.divide(np.nansum(x[train],0),counts,out=np.zeros(x.shape[1]),where=counts>0)
    x=np.where(np.isfinite(x),x,impute);mean=x[train].mean(0);sd=x[train].std(0);sd=np.where(sd==0,1.,sd)
    return (x-mean)/sd,dict(imputer=impute,mean=mean,scale=sd)


def direct(x,base,groups,train,y,weight):
    cols=np.r_[np.arange(base),base+np.array([2*g+j for g in groups for j in [0,1]],int)]
    weight=weight*len(weight)/weight.sum()
    if len(cols):
        model=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(x[train][:,cols],2*y[train]-1,sample_weight=weight)
        score=model.predict(x[:,cols])
    else:score=np.full(len(x),np.average(2*y[train]-1,weights=weight))
    return expit(score)


def main():
    binding=jr(ROOT/'INPUT_FREEZE_PRIVATE.json');assert binding['source']==source_binding()
    registry=read(ROOT/'INPUT_REGISTRY_PRIVATE.pkl');frozen=jr(ROOT/'CHOICES_FROZEN_PRIVATE.json');seal=jr(ROOT/'ASSESSMENT_SCORES_SEAL_PRIVATE.json')
    assert registry['binding']==frozen['binding']==seal['binding']==binding
    assert sha(ROOT/'CHOICES_FROZEN_PRIVATE.json')==seal['choice_freeze_sha256']
    p=primitives();rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');pool=read(OLDROOT/'POOL_PRIVATE.pkl');m=read(QUAL/'MEASUREMENTS_PRIVATE.pkl')
    rawreg=read(RAWROOT/'ENCODER_REGISTRY_PRIVATE.pkl');search_replayed=assessment_replayed=0;maximum=0.;max_prob=0.;plans=[]
    # The qualifier's independent scalar implementation is label-free and uses no selectors.
    import importlib.util
    spec=importlib.util.spec_from_file_location('nonlocal6_qualification_independent',QUALCODE/'verify_qualification.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    for entry in registry['entries']:
        assert sha(entry['path'])==entry['sha256'];view=read(Path(entry['path']));f=view['fold'];s=view['screen'];roles=view['roles'];pid=view['patient'];ix=view['archive_indices']
        ident=fit_identity_bank(f,rec);assert np.array_equal(ix,ident['archive_indices'])
        assert np.array_equal(view['patient'],ident['patient']) and np.array_equal(view['channel'],ident['channel'])
        actual=next(e for e in rawreg['entries'] if e['fold']==f and e['screen']==s);assert sha(actual['path'])==actual['sha256']==view['raw_entry_sha256']
        raw=read(Path(actual['path']));np.testing.assert_array_equal(raw['raw'],view['raw']);assert raw['checkpoint_sha256']==view['raw_checkpoint_sha256']
        np.testing.assert_allclose(view['new12'],independent.independently_z(m['values'][ix],pid),atol=1e-11,rtol=1e-11,equal_nan=True)
        np.testing.assert_allclose(view['old14'],independent.independently_z(pool['summary'][ix,:14].astype(float),pid),atol=1e-11,rtol=1e-11,equal_nan=True)
        qrole=next(r for r in m['roles'] if r['fold']==f and r['screen']==s)
        sets=[set(pid[a]) for a in roles.values()]
        assert all(sets[a].isdisjoint(sets[b]) for a in range(len(sets)) for b in range(a))
        assert set.union(*sets)==set(ident['patient']) and set(view['role_labels'])=={'train','cal','choice'}
        y=np.full(len(pid),np.nan)
        for role in roles:
            assert np.array_equal(ix[roles[role]],qrole['indices'][role])
            if role!='held':
                np.testing.assert_array_equal(view['role_labels'][role],ident['own_fit_ez'][roles[role]])
                y[roles[role]]=view['role_labels'][role]
        if s:assert np.isnan(y[roles['held']]).all()
        tr=roles['train'];cal=roles['cal'];ch=roles['choice']
        for context in CONTEXTS:
            cell=next(c for c in frozen['cells'] if c['fold']==f and c['screen']==s and c['context']==context)
            assert sha(cell['path'])==cell['sha256'];saved=read(Path(cell['path']));x,base=context_input(view,context)
            prepared,pre=independent_prepare(x,tr)
            for key in pre:np.testing.assert_array_equal(pre[key],saved['pre'][key])
            assert len(saved['rows'])==2**len(view['eligible'])
            replay=[];weight=p.weights(y[tr].astype(int),pid[tr])
            for row in saved['rows']:
                prob=direct(prepared,base,row['groups'],tr,y,weight);theta=p.threshold(y[cal].astype(int),prob[cal],pid[cal])
                rows=p.metric_rows(y[ch].astype(int),prob[ch],pid[ch],view['channel'][ch],view['center'][ch],theta)
                values={metric:float(np.nanmean([r[metric] for r in rows])) for metric in METRICS}
                u=float(np.mean([values[k] for k in UTILITY]));sources=[]
                for c in sorted(set(view['center'][ch])):
                    own=[r for r in rows if r['center']==c];sources.append(np.mean([np.nanmean([r[k] for r in own]) for k in UTILITY]))
                sd=float(np.std(sources,ddof=1)) if len(sources)>1 else 0.
                for metric in METRICS:assert abs(values[metric]-row[metric])<=1e-12
                error=abs(u-.1*sd-row['choice_score']);assert error<=1e-12 and theta==row['threshold'];maximum=max(maximum,error)
                replay.append(dict(groups=row['groups'],k=len(row['groups']),score=u-.1*sd));search_replayed+=1
            best=max(r['score'] for r in replay);k=min(r['k'] for r in replay if r['score']>=best-.005)
            selected=min([r for r in replay if r['k']==k],key=lambda r:(-r['score'],tuple(r['groups'])))
            assert selected['groups']==saved['arms']['SELECTED']
            alt=[r for r in replay if r['k']==k and r['groups']!=selected['groups']]
            expected=min(alt,key=lambda r:(-r['score'],tuple(r['groups'])))['groups'] if alt else selected['groups']
            assert expected==saved['arms']['BEST_ALTERNATIVE']
            rng=np.random.default_rng(42+1009*f+43*s+100000*k)
            for arm in ['RANDOM_A','RANDOM_B']:assert sorted(rng.choice(view['eligible'],size=k,replace=False).tolist())==saved['arms'][arm]
            for kk in range(len(view['eligible'])+1):
                expected=min([r for r in replay if r['k']==kk],key=lambda r:(-r['score'],tuple(r['groups'])))['groups']
                assert saved['arms'][f'K{kk}']==expected
            plans.append(dict(fold=f,screen=s,context=context,k=k))
            if not s:continue
            assessed=next(c for c in seal['cells'] if c['fold']==f and c['screen']==s and c['context']==context)
            assert sha(assessed['path'])==assessed['sha256'];result=read(Path(assessed['path']))
            assert result['choice_sha256']==cell['sha256'];refit=np.r_[tr,ch];xx,pp=independent_prepare(x,refit)
            for key in pp:np.testing.assert_array_equal(pp[key],result['pre'][key])
            for key,model in result['models'].items():
                prob=direct(xx,base,model['groups'],refit,y,p.weights(y[refit].astype(int),pid[refit]))
                err=float(np.max(np.abs(prob[roles['held']]-model['HELD_probability'])));max_prob=max(max_prob,err);assert err<=1e-12
                np.testing.assert_allclose(prob[cal],model['calibration_probability'],atol=1e-12,rtol=0)
                assert p.threshold(y[cal].astype(int),prob[cal],pid[cal])==model['threshold'];assessment_replayed+=1
            assert result['aliases']=={arm:','.join(map(str,gg)) for arm,gg in saved['arms'].items()}
        print('ALL_DENSE_HEADS_AND_SELECTIONS_REPLAYED',f,s,search_replayed,assessment_replayed,flush=True)
    report=jr(PUB/'REPORT_STATUS.json');assert report['actual_search_fits']==search_replayed and report['actual_assessment_refits']==assessment_replayed
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal']
    write(PUB/'INDEPENDENT_EXPERIMENT_VERIFICATION.json',dict(status='PASS',search_fits_replayed=search_replayed,
        assessment_refits_replayed=assessment_replayed,context_choices_replayed=len(plans),maximum_choice_score_error=maximum,
        maximum_assessment_probability_error=max_prob,qualified_old_and_new_values_independently_normalized=True,
        all_roles_and_train_only_scalers_verified=True,HELD_labels_absent_in_selector_inputs=True,
        canonical_API_replay_not_independent_numerical_precision_test=True,
        independent_patient_metric_rows=report['logical_patient_metric_rows'],new_Outer_access=False,source_sha256=sha(HERE/'verify_experiment.py')))
    print('COMPLETE_5120_SEARCH_AND_ALL_ASSESSMENT_REPLAY_PASS',flush=True)


if __name__=='__main__':main()
