"""Actual reused/new checkpoint replay and complete 2x2 contrast verification."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model,collapsed_state
from admit import evaluate
from train_internal import baseline,reused_baseline,first_gate,selection,paired,parameters,penalized

def verify_metrics(d,ix,logit,theta,part):
    for r in part:
        j=d['patient'][ix]==r['patient'];y=1-d['y'][ix][j];pn=probability(logit[j]);p=1-pn.astype(float)
        pred=pn<theta;ch=d['channel'][ix][j]
        order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if y[i])
        independent=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),f1=f1_score(y,pred,zero_division=0),
            nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),accuracy=accuracy_score(y,pred),auroc=roc_auc_score(y,p),
            auprc=average_precision_score(y,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
        for m,v in independent.items():assert abs(r[m]-v)<1e-12



def factorial_frames(first):
    keys=['patient','center','fold','screen']
    cells={a:first[first.arm==a].set_index(keys)[METRICS].sort_index() for a in FAMILY}
    assert all(cells['SHORT_MEAN'].index.equals(z.index) for z in cells.values())
    values=factorial_values(*(cells[a] for a in FAMILY))
    return {name:value.reset_index() for name,value in values.items()},cells


def tables(frame):
    first=frame[frame.screen==1]
    first.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'FIRST_PATH_MAIN.csv',index=False)
    ff=first.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUB/'FIRST_PATH_FOLD.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/'FIRST_PATH_FOLD_SD.csv',index=False)
    pairs=[(a,b) for a in NOVEL for b in REFERENCES]+[
        ('SHORT_ENERGY','SHORT_MEAN'),('LONG_MEAN','SHORT_MEAN'),('LONG_ENERGY','SHORT_MEAN'),
        ('LONG_MEAN','SHORT_ENERGY'),('LONG_ENERGY','SHORT_ENERGY'),('LONG_ENERGY','LONG_MEAN')]
    source=[];absolute=[];intervals=[];factor=[]
    for (arm,center),g in first.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in first.groupby('arm'):
        assert len(g)==86 and g.patient.nunique()==59
        absolute.extend([dict(arm=arm,**q) for q in parent.cluster_interval(g.to_dict('records'))])
    for left,right in pairs:
        diff=paired(first[first.arm.isin([left,right])],left,right)
        intervals.extend([dict(left=left,right=right,**q) for q in parent.cluster_interval(diff.to_dict('records'))])
    factors,cells=factorial_frames(first)
    for name,g in factors.items():factor.extend([dict(contrast=name,**q) for q in parent.cluster_interval(g.to_dict('records'))])
    pd.DataFrame(source).to_csv(PUB/'FIRST_PATH_SOURCE.csv',index=False)
    pd.DataFrame(absolute).to_csv(PUB/'FIRST_PATH_ABSOLUTE_CI.csv',index=False)
    pd.DataFrame(intervals).to_csv(PUB/'FIRST_PATH_PAIRED_CI.csv',index=False)
    pd.DataFrame(factor).to_csv(PUB/'FACTORIAL_CONTRASTS.csv',index=False)
    coverage=[dict(arm=arm,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g),unique_patients=g.patient.nunique())
              for arm,g in first.groupby('arm') for m in METRICS]
    pd.DataFrame(coverage).to_csv(PUB/'FIRST_PATH_METRIC_COVERAGE.csv',index=False)
    maximum=0.;checks=0
    def verify_ci(reported,g):
        nonlocal maximum,checks
        ids=sorted(g.patient.unique());sums=np.stack([g[g.patient==pid][METRICS].sum().to_numpy(float) for pid in ids])
        counts=np.asarray([int((g.patient==pid).sum()) for pid in ids])
        draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
        samples=sums[draw].sum(1)/counts[draw].sum(1)[:,None];ci=np.quantile(samples,[.025,.975],axis=0)
        for j,m in enumerate(METRICS):
            row=reported[reported.metric==m].iloc[0]
            error=max(abs(float(row['mean'])-float(g[m].mean())),abs(float(row.low)-ci[0,j]),abs(float(row.high)-ci[1,j]))
            maximum=max(maximum,error);assert error<1e-12 and int(row.draws)==10000 and int(row.unique_patients)==len(ids);checks+=1
    aa=pd.read_csv(PUB/'FIRST_PATH_ABSOLUTE_CI.csv');pp=pd.read_csv(PUB/'FIRST_PATH_PAIRED_CI.csv');fc=pd.read_csv(PUB/'FACTORIAL_CONTRASTS.csv')
    for arm,g in first.groupby('arm'):verify_ci(aa[aa.arm==arm],g)
    for left,right in pairs:
        diff=paired(first[first.arm.isin([left,right])],left,right)
        verify_ci(pp[(pp.left==left)&(pp.right==right)],diff)
    # Independent coefficient form checks the definitions as well as intervals.
    coefficients={'kernel_main':[-.5,-.5,.5,.5],'energy_main':[-.5,.5,-.5,.5],'interaction':[1.,-1.,-1.,1.]}
    stack=np.stack([cells[a].to_numpy(float) for a in FAMILY],axis=1);ids=cells['SHORT_MEAN'].index.get_level_values('patient')
    for name,weights in coefficients.items():
        independent=np.einsum('afm,f->am',stack,np.asarray(weights))
        g=pd.DataFrame(independent,columns=METRICS);g.insert(0,'patient',ids)
        assert np.max(np.abs(independent-factors[name][METRICS].to_numpy(float)))<1e-12
        verify_ci(fc[fc.contrast==name],g)
    assert checks==261
    return checks,maximum


def verify_choice_intervals(frame,gates):
    maximum=0.;checks=0
    for gate in gates:
        fold,arm=gate['fold'],gate['arm'];g=frame[(frame.fold==fold)&(frame.arm.isin([arm]+REFERENCES[:2]+['SHORT_MEAN']))]
        for contrast in gate['contrasts']:
            diff=paired(g[g.arm.isin([arm,contrast['reference']])],arm,contrast['reference'])
            ids=sorted(diff.patient.unique());sums=np.array([diff[diff.patient==p].macro_f1.sum() for p in ids])
            n=np.array([len(diff[diff.patient==p]) for p in ids]);draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
            lo,hi=np.quantile(sums[draw].sum(1)/n[draw].sum(1),[.025,.975])
            error=max(abs(float(lo)-contrast['paired_Macro_low']),abs(float(hi)-contrast['paired_Macro_high']))
            maximum=max(maximum,error);assert error<1e-12;checks+=1
    return checks,maximum



def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_EARLY_FIR_INTERNAL_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    assert seal['input']==binding and status['score_seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    cc={(r['fold'],r['screen']):r for r in contexts()};expected={(f,1,a) for f in range(1,6) for a in ARMS}
    for race in seal['races']:
        if race['advanced']:
            for screen in [2,3]:expected.update((race['fold'],screen,a) for a in reference_names(cc[(race['fold'],screen)])+FAMILY)
    actual=pd.read_csv(PUB/'INTERNAL_CELL_RESULTS.csv')
    assert set(zip(actual.fold,actual.screen,actual.arm))==expected and len(actual)==len(expected)
    store=RawStore();rows=[];checks=[];maximum=0.
    for fold,screen,arm in sorted(expected):
        ctx=cc[(fold,screen)];d=ctx['data'];cal,held=ctx['roles']['cal'],ctx['roles']['held']
        if arm in REFERENCES:
            summary,part=baseline(ctx,arm);logit,_=reference(ctx,arm);verify_metrics(d,held,logit,summary['threshold'],part)
        else:
            record=next(r for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(fold,screen,arm))
            hp,bp=Path(record['hp']),Path(record['bp']);assert sha(hp)==record['sha256'] and sha(bp)==record['checkpoint_sha256']
            q=read(hp);state=torch.load(bp,map_location='cpu',weights_only=False)
            assert state['binding']==q['binding'] and state['epoch']>=1 and sha(bp)==q['summary']['checkpoint_sha256']
            reuse=record['kind']=='REUSED_EXACT_SHORT_MEAN'
            if reuse:
                source=ctx['reused_short_mean'];assert arm=='SHORT_MEAN' and source is not None
                assert hp==source['hp'] and bp==source['bp'] and q['binding']['arm']=='FRESH_BASE'
            else:
                assert record['kind']=='NEW_FIT' and q['binding']['input']==binding and q['binding']['arm']==arm
                assert hp==ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm/'HELD_PRIVATE.pkl'
            assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
            model=Model(arm,fold).cuda();assert state_hash(model.state_dict())==q['binding']['initial_state_sha256']
            core=state_hash(collapsed_state(model));model.load_state_dict(state['model'])
            cl=evaluate(model,d,cal,store);hh=evaluate(model,d,held,store)
            error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hh-q['logit_nez']))));maximum=max(maximum,error);assert error<=1e-6
            grid=np.round(np.arange(0,1.0001,.005),3)
            assert np.array_equal(probability(cl)[:,None]<grid,probability(q['cal_logit_nez'])[:,None]<grid)
            assert calibrate(d,cal,cl)==state['selection'];part=held_rows(ctx,hh,state['selection']['threshold'],arm)
            verify_metrics(d,held,hh,state['selection']['threshold'],part)
            summary=reused_baseline(ctx)[0] if reuse else q['summary']
            checks.append(dict(fold=fold,screen=screen,arm=arm,execution_kind=record['kind'],maximum_replay_error=error,
                selected_epoch=state['epoch'],initial_state_sha256=q['binding']['initial_state_sha256'],collapsed_core_initial_sha256=core,all201_CAL_grid_decisions_exact=True))
            del model;gc.collect();torch.cuda.empty_cache();print('EARLY_FIR_CELL_INDEPENDENTLY_VERIFIED',fold,screen,arm,record['kind'],flush=True)
        pub=actual[(actual.fold==fold)&(actual.screen==screen)&(actual.arm==arm)].iloc[0]
        for m in METRICS:assert abs(float(pub[m])-float(summary[m]))<1e-12
        assert int(pub.patients)==len(part) and abs(float(pub.threshold)-summary['threshold'])<1e-12
        assert int(pub.parameters)==parameters(arm);rows.extend(part)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    fresh=sum(r['execution_kind']=='NEW_FIT' for r in checks);reused=len(checks)-fresh
    assert fresh==status['actual_neural_fits'] and reused==status['reused_control_cells'] and len(checks)==status['family_model_cells']
    assert len(pd.read_csv(PUB/'TRAIN_HISTORY.csv'))==status['epochs']
    for fold in range(1,6):assert len({q['collapsed_core_initial_sha256'] for q in checks if q['fold']==fold})==1
    ci_checks,ci_error=tables(frame);choices=[];gates=[];raced=[]
    for race in seal['races']:
        fold=race['fold'];first=[]
        for arm in ARMS:
            z=frame[(frame.fold==fold)&(frame.screen==1)&(frame.arm==arm)]
            first.append(dict(arm=arm,parameters=parameters(arm),penalized_macro=penalized(z.to_dict('records')),
                macro_f1=float(z.macro_f1.mean()),auroc=float(z.auroc.mean())))
        check=first_gate(first);assert check['chosen']==race['chosen'] and check['advanced']==race['advanced']
        for key in ['Macro_gain','AUROC_gain']:assert abs(check[key]-race[key])<1e-12
        assert race['promoted']==(FAMILY if check['advanced'] else [])
        choice,gg=selection(frame,race);choices.append(choice);gates.extend(gg)
        if race['advanced']:
            for arm in ARMS:
                z=frame[(frame.fold==fold)&(frame.arm==arm)]
                if set(z.screen)!={1,2,3}:continue
                raced.append(dict(fold=fold,arm=arm,appearances=len(z),unique_patients=z.patient.nunique(),**{m:float(z[m].mean()) for m in METRICS}))
    gate_checks,gate_error=verify_choice_intervals(frame,gates)
    pd.DataFrame(raced,columns=['fold','arm','appearances','unique_patients']+METRICS).to_csv(PUB/'RACED_PATH_RESULTS.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['scores','eligible_arms','fallback_scores']} for r in choices]).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    write(PUB/'REPLICATION_ELIGIBILITY.json',dict(status='RECOMPUTED_WITH_INDEPENDENT_INTERVAL_CHECK',gates=gates))
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices,gates=gates))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,actual_neural_fits=fresh,reused_control_cells=reused,
        family_model_cells=len(checks),logical_cells=len(expected),patient_metric_rows=len(frame),maximum_replay_error=maximum,
        all9_patient_metrics_independent=True,all_races_recomputed=True,all4_collapsed_initial_core_states_exact_per_fold=True,
        reused_actual_model_predictions_verified=True,public_cell_summaries_verified=True,independent_CI_metric_checks=ci_checks,
        maximum_bootstrap_difference=ci_error,replication_interval_checks=gate_checks,maximum_replication_interval_error=gate_error,
        ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_EARLY_FIR_INTERNAL',actual_neural_fits=fresh,reused_control_cells=reused,
        family_model_cells=len(checks),epochs=status['epochs'],logical_cells=len(expected),
        advanced_folds=[r['fold'] for r in choices if r['advanced']],fresh_qualified_folds=[r['fold'] for r in choices if r['fresh_eligible']],
        final_stage_warranted=any(r['fresh_eligible'] for r in choices),new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('EARLY_FIR_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
