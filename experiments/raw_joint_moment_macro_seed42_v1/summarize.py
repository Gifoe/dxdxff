"""Independent retained-state replay, metrics, clustered CIs and legal choices."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model
from admit import evaluate
from train_internal import baseline,choose,penalized,parameters


def tables(frame):
    first=frame[frame.screen==1];first.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'FIRST_PATH_MAIN.csv',index=False)
    ff=first.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUB/'FIRST_PATH_FOLD.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/'FIRST_PATH_FOLD_SD.csv',index=False)
    source=[];absolute=[];paired=[]
    for (arm,center),g in first.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in first.groupby('arm'):absolute.extend([dict(arm=arm,**r) for r in parent.cluster_interval(g.to_dict('records'))])
    pairs=[(a,b) for a in TRAINED for b in REFERENCES]+[('PRE_JOINT','PRE_MEAN_DUP'),('PRE_JOINT','PRE_ENERGY_DUP'),('PRE_ENERGY_DUP','PRE_MEAN_DUP')]
    for left,right in pairs:
        q=first[first.arm==left].merge(first[first.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in parent.cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(source).to_csv(PUB/'FIRST_PATH_SOURCE.csv',index=False);pd.DataFrame(absolute).to_csv(PUB/'FIRST_PATH_ABSOLUTE_CI.csv',index=False)
    pd.DataFrame(paired).to_csv(PUB/'FIRST_PATH_PAIRED_CI.csv',index=False)
    coverage=[dict(arm=arm,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g),unique_patients=g.patient.nunique())
              for arm,g in first.groupby('arm') for m in METRICS]
    pd.DataFrame(coverage).to_csv(PUB/'FIRST_PATH_METRIC_COVERAGE.csv',index=False)
    maximum=0.;checks=0
    def verify_ci(reported,g):
        nonlocal maximum,checks
        ids=sorted(g.patient.unique());sums=np.stack([g[g.patient==pid][METRICS].sum().to_numpy(float) for pid in ids])
        counts=np.asarray([int((g.patient==pid).sum()) for pid in ids]);draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
        samples=sums[draw].sum(1)/counts[draw].sum(1)[:,None];ci=np.quantile(samples,[.025,.975],axis=0)
        for j,m in enumerate(METRICS):
            r=reported[reported.metric==m].iloc[0];error=max(abs(float(r['mean'])-float(g[m].mean())),abs(float(r.low)-ci[0,j]),abs(float(r.high)-ci[1,j]))
            maximum=max(maximum,error);assert error<1e-12 and int(r.draws)==10000 and int(r.unique_patients)==len(ids);checks+=1
    aa=pd.read_csv(PUB/'FIRST_PATH_ABSOLUTE_CI.csv');pp=pd.read_csv(PUB/'FIRST_PATH_PAIRED_CI.csv')
    for arm,g in first.groupby('arm'):verify_ci(aa[aa.arm==arm],g)
    for left,right in pairs:
        q=first[first.arm==left].merge(first[first.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        g=pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))
        verify_ci(pp[(pp.left==left)&(pp.right==right)],g)
    assert checks==126
    return checks,maximum


def verify_metrics(d,ix,logit,theta,part):
    for r in part:
        j=d['patient'][ix]==r['patient'];y=1-d['y'][ix][j];pn=probability(logit[j]);p=1-pn.astype(float);pred=pn<theta;ch=d['channel'][ix][j]
        order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if y[i])
        independent=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),f1=f1_score(y,pred,zero_division=0),nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),
            accuracy=accuracy_score(y,pred),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
        for m,v in independent.items():assert abs(r[m]-v)<1e-12


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_JOINT_RAW_INTERNAL_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding and status['score_seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    cc={(r['fold'],r['screen']):r for r in contexts()};expected={(f,1,a) for f in range(1,6) for a in ARMS}
    for race in seal['races']:
        if race['advanced']:expected.update((race['fold'],s,a) for s in [2,3] for a in ARMS)
    actual=pd.read_csv(PUB/'INTERNAL_CELL_RESULTS.csv');assert set(zip(actual.fold,actual.screen,actual.arm))==expected and len(actual)==len(expected)
    store=RawStore();rows=[];checks=[];maximum=0.
    for fold,screen,arm in sorted(expected):
        ctx=cc[(fold,screen)];d=ctx['data'];cal,held=ctx['roles']['cal'],ctx['roles']['held']
        if arm in REFERENCES:
            summary,part=baseline(ctx,arm);x=ctx['native_logits'] if arm=='NATIVE' else ctx['pre_logits']
            verify_metrics(d,held,x,summary['threshold'],part);rows.extend(part);continue
        cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;hp,bp=cell/'HELD_PRIVATE.pkl',cell/'BEST_PRIVATE.pt'
        expectedsha=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(fold,screen,arm));assert sha(hp)==expectedsha
        q=read(hp);state=torch.load(bp,map_location='cpu',weights_only=False)
        assert state['binding']==q['binding'] and q['binding']['input']==binding and state['epoch']>=1 and sha(bp)==q['summary']['checkpoint_sha256']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
        model=Model(arm,fold).cuda();assert state_hash(model.state_dict())==q['binding']['initial_state_sha256']
        model.load_state_dict(state['model']);cl=evaluate(model,d,cal,store);hh=evaluate(model,d,held,store)
        error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hh-q['logit_nez']))));maximum=max(maximum,error);assert error<=1e-6
        grid=np.round(np.arange(0,1.0001,.005),3)
        assert np.array_equal(probability(cl)[:,None]<grid,probability(q['cal_logit_nez'])[:,None]<grid)
        assert calibrate(d,cal,cl)==state['selection'];part=held_rows(ctx,hh,state['selection']['threshold'],arm)
        verify_metrics(d,held,hh,state['selection']['threshold'],part)
        rows.extend(part);checks.append(dict(fold=fold,screen=screen,arm=arm,maximum_replay_error=error,
            selected_epoch=state['epoch'],initial_state_sha256=q['binding']['initial_state_sha256'],all201_CAL_grid_decisions_exact=True))
        del model;gc.collect();torch.cuda.empty_cache();print('JOINT_RAW_CELL_INDEPENDENTLY_VERIFIED',fold,screen,arm,flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    for fold in range(1,6):assert len({q['initial_state_sha256'] for q in checks if q['fold']==fold})==1
    ci_checks,ci_error=tables(frame);choices=[];raced=[]
    for race in seal['races']:
        fold=race['fold'];first=[]
        for arm in ARMS:
            g=frame[(frame.fold==fold)&(frame.screen==1)&(frame.arm==arm)]
            first.append(dict(arm=arm,parameters=parameters(arm),penalized_macro=penalized(g.to_dict('records')),macro_f1=float(g.macro_f1.mean()),auroc=float(g.auroc.mean())))
        candidate=choose(first[2:]);advanced=candidate['macro_f1']>=max(r['macro_f1'] for r in first[:2])+.01 and candidate['auroc']>=max(r['auroc'] for r in first[:2])-.01
        assert candidate['arm']==race['chosen'] and advanced==race['advanced'] and race['promoted']==(TRAINED if advanced else [])
        selected='NATIVE';scores=[]
        if advanced:
            for arm in ARMS:
                g=frame[(frame.fold==fold)&(frame.arm==arm)];assert set(g.screen)=={1,2,3}
                scores.append(dict(arm=arm,parameters=parameters(arm),penalized_macro=penalized(g.to_dict('records'))))
                raced.append(dict(fold=fold,arm=arm,appearances=len(g),unique_patients=g.patient.nunique(),**{m:float(g[m].mean()) for m in METRICS}))
            selected=choose(scores)['arm']
        choices.append(dict(fold=fold,advanced=advanced,first_candidate=race['chosen'],selected_arm=selected,scores=scores))
    pd.DataFrame(raced,columns=['fold','arm','appearances','unique_patients']+METRICS).to_csv(PUB/'RACED_PATH_RESULTS.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='scores'} for r in choices]).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,actual_neural_fits=len(checks),logical_cells=len(expected),patient_metric_rows=len(frame),
        maximum_replay_error=maximum,all9_patient_metrics_independent=True,all_races_recomputed=True,all3_random_initial_states_exact_per_fold=True,
        independent_CI_metric_checks=ci_checks,maximum_bootstrap_difference=ci_error,
        ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_JOINT_RAW_INTERNAL',actual_neural_fits=len(checks),epochs=status['epochs'],logical_cells=len(expected),
        advanced_folds=[r['fold'] for r in choices if r['advanced']],final_stage_warranted=any(r['advanced'] for r in choices),new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('JOINT_RAW_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
