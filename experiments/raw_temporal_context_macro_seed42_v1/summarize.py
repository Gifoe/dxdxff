"""Independent selected-state/role/metric replay, bounded-race and policy audit."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model
from admit import evaluate
from train_internal import baseline,penalized,choose


def summaries(frame,prefix):
    frame.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/(prefix+'_MAIN.csv'),index=False)
    fm=frame.groupby(['fold','arm'])[METRICS].mean().reset_index();fm.to_csv(PUB/(prefix+'_FOLD.csv'),index=False)
    fm.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/(prefix+'_FOLD_SD.csv'),index=False)
    source=[];absolute=[];paired=[]
    for (arm,center),g in frame.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in frame.groupby('arm'):absolute.extend([dict(arm=arm,**r) for r in parent.cluster_interval(g.to_dict('records'))])
    pairs=[(a,b) for a in TRAINED for b in ARMS[:2]]+[(a,'PRE_TUNE') for a in TRAINED[1:]]+[('TEMPORAL_CONTEXT','CHANNEL_MLP')]
    for left,right in pairs:
        q=frame[frame.arm==left].merge(frame[frame.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in parent.cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(source).to_csv(PUB/(prefix+'_SOURCE.csv'),index=False);pd.DataFrame(absolute).to_csv(PUB/(prefix+'_ABSOLUTE_CI.csv'),index=False)
    pd.DataFrame(paired).to_csv(PUB/(prefix+'_PAIRED_CI.csv'),index=False)


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_TEMPORAL_INTERNAL_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding and status['score_seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    cc={(r['fold'],r['screen']):r for r in contexts()};expected={(f,1,a) for f in range(1,6) for a in ARMS}
    for race in seal['races']:
        if race['advanced']:expected.update((race['fold'],s,a) for s in [2,3] for a in ARMS)
    actual=pd.read_csv(PUB/'INTERNAL_CELL_RESULTS.csv');assert set(zip(actual.fold,actual.screen,actual.arm))==expected and len(actual)==len(expected)
    store=RawStore();rows=[];checks=[];maximum=0.
    for fold,screen,arm in sorted(expected):
        ctx=cc[(fold,screen)];d=ctx['data'];cal,held=ctx['roles']['cal'],ctx['roles']['held']
        if arm in ARMS[:2]:_,part=baseline(ctx,arm);rows.extend(part);continue
        cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;hp,bp=cell/'HELD_PRIVATE.pkl',cell/'BEST_PRIVATE.pt'
        sh=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(fold,screen,arm));assert sha(hp)==sh
        q=read(hp);best=torch.load(bp,map_location='cpu',weights_only=False)
        assert q['binding']==best['binding'] and q['binding']['input']==binding and sha(bp)==q['summary']['checkpoint_sha256']
        assert q['binding']['source_matrix']==ctx['pre_matrix_sha256'] and q['binding']['source_body']==ctx['stored']['checkpoint_sha256']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
        m=Model(arm,fold,ctx['reference_state'],ctx['matrix']).cuda();assert state_hash(m.state_dict())==q['binding']['initial_state_sha256']
        m.load_state_dict(best['model']);cl=evaluate(m,d,cal,store);hh=evaluate(m,d,held,store)
        error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hh-q['logit_nez']))));maximum=max(maximum,error);assert error<=1e-6
        assert calibrate(d,cal,cl)==best['selection'];part=held_rows(ctx,hh,best['selection']['threshold'],arm)
        for r in part:
            j=d['patient'][held]==r['patient'];y=1-d['y'][held][j];pn=probability(hh[j]);p=1-pn.astype(float);pred=pn<best['selection']['threshold']
            order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(d['channel'][held][j][k])));first=next(k for k,i in enumerate(order) if y[i])
            metrics=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),f1=f1_score(y,pred,zero_division=0),nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),
                accuracy=accuracy_score(y,pred),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
            for k,v in metrics.items():assert abs(r[k]-v)<1e-12
        rows.extend(part);checks.append(dict(fold=fold,screen=screen,arm=arm,maximum_replay_error=error,selected_epoch=best['epoch']))
        del m;gc.collect();torch.cuda.empty_cache();print('RAW_TEMPORAL_INDEPENDENT_CELL_PASS',fold,screen,arm,flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    summaries(frame[frame.screen==1],'FIRST_PATH');choices=[];raced=[]
    for race in seal['races']:
        fold=race['fold'];first=[]
        for arm in ARMS:
            g=frame[(frame.fold==fold)&(frame.screen==1)&(frame.arm==arm)];params=12529 if arm=='NATIVE' else (13105 if arm in ['FROZEN_PRE','PRE_TUNE'] else 14089)
            first.append(dict(arm=arm,parameters=params,penalized_macro=penalized(g.to_dict('records')),macro_f1=float(g.macro_f1.mean()),auroc=float(g.auroc.mean())))
        best=choose(first[2:]);advanced=best['macro_f1']>=max(r['macro_f1'] for r in first[:2])+.01 and best['auroc']>=max(r['auroc'] for r in first[:2])-.01
        assert best['arm']==race['chosen'] and advanced==race['advanced']
        selected='NATIVE';scores=[]
        if advanced:
            for arm in ARMS:
                g=frame[(frame.fold==fold)&(frame.arm==arm)];assert set(g.screen)=={1,2,3}
                params=next(r['parameters'] for r in first if r['arm']==arm)
                scores.append(dict(arm=arm,parameters=params,penalized_macro=penalized(g.to_dict('records'))))
                raced.append(dict(fold=fold,arm=arm,appearances=len(g),unique_patients=g.patient.nunique(),**{m:float(g[m].mean()) for m in METRICS}))
            selected=choose(scores)['arm']
        choices.append(dict(fold=fold,advanced=advanced,first_path_candidate=race['chosen'],selected_arm=selected,scores=scores))
    pd.DataFrame(raced,columns=['fold','arm','appearances','unique_patients']+METRICS).to_csv(PUB/'RACED_PATH_RESULTS.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='scores'} for r in choices]).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,actual_neural_fits=len(checks),logical_cells=len(expected),
        patient_metric_rows=len(frame),maximum_replay_error=maximum,all9_patient_metrics_independent=True,all_race_decisions_recomputed=True,
        ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_TEMPORAL_CONTEXT_INTERNAL',actual_neural_fits=len(checks),epochs=status['epochs'],
        logical_cells=len(expected),advanced_folds=[r['fold'] for r in choices if r['advanced']],new_Outer_passes=0,goal_established=False,
        final_stage_warranted=any(r['advanced'] for r in choices),independent_confirmation=False))
    print('RAW_TEMPORAL_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
