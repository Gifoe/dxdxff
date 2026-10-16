"""Reconstruct selected states, factorial contrasts, races and ownFIT choices."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model
from admit import evaluate
from train_internal import baseline,choose,penalized,promoted


def tables(frame):
    first=frame[frame.screen==1];first.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'FIRST_PATH_MAIN.csv',index=False)
    ff=first.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUB/'FIRST_PATH_FOLD.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/'FIRST_PATH_FOLD_SD.csv',index=False)
    source=[];absolute=[];paired=[]
    for (arm,center),g in first.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in first.groupby('arm'):absolute.extend([dict(arm=arm,**r) for r in parent.cluster_interval(g.to_dict('records'))])
    pairs=[(a,b) for a in TRAINED for b in REFERENCES]+[('WINDOW_LOGENERGY','WINDOW_MEAN'),('PRE_LOGENERGY','PRE_MEAN'),('PRE_MEAN','WINDOW_MEAN'),('PRE_LOGENERGY','WINDOW_LOGENERGY')]
    for left,right in pairs:
        q=first[first.arm==left].merge(first[first.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in parent.cluster_interval(diff.to_dict('records'))])
    interaction=first[first.arm=='PRE_LOGENERGY'].set_index(['patient','center','fold','screen'])[METRICS].copy()
    for arm,sign in [('PRE_MEAN',-1),('WINDOW_LOGENERGY',-1),('WINDOW_MEAN',1)]:
        interaction=interaction+sign*first[first.arm==arm].set_index(['patient','center','fold','screen'])[METRICS]
    it=interaction.reset_index();pd.DataFrame(parent.cluster_interval(it.to_dict('records'))).to_csv(PUB/'FIRST_PATH_FACTORIAL_INTERACTION_CI.csv',index=False)
    pd.DataFrame(source).to_csv(PUB/'FIRST_PATH_SOURCE.csv',index=False);pd.DataFrame(absolute).to_csv(PUB/'FIRST_PATH_ABSOLUTE_CI.csv',index=False)
    pd.DataFrame(paired).to_csv(PUB/'FIRST_PATH_PAIRED_CI.csv',index=False)


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_FRESH_ENERGY_INTERNAL_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding and status['score_seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    cc={(r['fold'],r['screen']):r for r in contexts()};expected={(f,1,a) for f in range(1,6) for a in ARMS}
    for race in seal['races']:
        if race['advanced']:expected.update((race['fold'],s,a) for s in [2,3] for a in REFERENCES+race['promoted'])
    actual=pd.read_csv(PUB/'INTERNAL_CELL_RESULTS.csv');assert set(zip(actual.fold,actual.screen,actual.arm))==expected and len(actual)==len(expected)
    store=RawStore();rows=[];checks=[];maximum=0.
    for fold,screen,arm in sorted(expected):
        ctx=cc[(fold,screen)];d=ctx['data'];cal,held=ctx['roles']['cal'],ctx['roles']['held']
        if arm in REFERENCES:_,part=baseline(ctx,arm);rows.extend(part);continue
        cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;hp,bp=cell/'HELD_PRIVATE.pkl',cell/'BEST_PRIVATE.pt'
        expectedsha=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(fold,screen,arm));assert sha(hp)==expectedsha
        q=read(hp);state=torch.load(bp,map_location='cpu',weights_only=False)
        assert state['binding']==q['binding'] and q['binding']['input']==binding and state['epoch']>=1 and sha(bp)==q['summary']['checkpoint_sha256']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
        model=Model(arm,fold).cuda();assert state_hash(model.state_dict())==q['binding']['initial_state_sha256']
        model.load_state_dict(state['model']);cl=evaluate(model,d,cal,store);hh=evaluate(model,d,held,store)
        error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hh-q['logit_nez']))));maximum=max(maximum,error);assert error<=1e-6
        assert calibrate(d,cal,cl)==state['selection'];part=held_rows(ctx,hh,state['selection']['threshold'],arm)
        for r in part:
            j=d['patient'][held]==r['patient'];y=1-d['y'][held][j];pn=probability(hh[j]);p=1-pn.astype(float);pred=pn<state['selection']['threshold'];ch=d['channel'][held][j]
            order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if y[i])
            independent=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),f1=f1_score(y,pred,zero_division=0),nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),
                accuracy=accuracy_score(y,pred),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
            for m,v in independent.items():assert abs(r[m]-v)<1e-12
        rows.extend(part);checks.append(dict(fold=fold,screen=screen,arm=arm,maximum_replay_error=error,selected_epoch=state['epoch'],initial_state_sha256=q['binding']['initial_state_sha256']))
        del model;gc.collect();torch.cuda.empty_cache();print('FRESH_RAW_CELL_INDEPENDENTLY_VERIFIED',fold,screen,arm,flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    tables(frame);choices=[];raced=[]
    for race in seal['races']:
        fold=race['fold'];first=[]
        for arm in ARMS:
            g=frame[(frame.fold==fold)&(frame.screen==1)&(frame.arm==arm)]
            first.append(dict(arm=arm,parameters=13105 if arm=='FROZEN_PRE' else 12529,penalized_macro=penalized(g.to_dict('records')),macro_f1=float(g.macro_f1.mean()),auroc=float(g.auroc.mean())))
        candidate=choose(first[2:]);advanced=candidate['macro_f1']>=max(r['macro_f1'] for r in first[:2])+.01 and candidate['auroc']>=max(r['auroc'] for r in first[:2])-.01
        assert candidate['arm']==race['chosen'] and advanced==race['advanced']
        assert race['promoted']==(promoted(candidate['arm']) if advanced else [])
        selected='NATIVE';scores=[]
        if advanced:
            for arm in REFERENCES+race['promoted']:
                g=frame[(frame.fold==fold)&(frame.arm==arm)];assert set(g.screen)=={1,2,3}
                scores.append(dict(arm=arm,parameters=13105 if arm=='FROZEN_PRE' else 12529,penalized_macro=penalized(g.to_dict('records'))))
                raced.append(dict(fold=fold,arm=arm,appearances=len(g),unique_patients=g.patient.nunique(),**{m:float(g[m].mean()) for m in METRICS}))
            selected=choose(scores)['arm']
        choices.append(dict(fold=fold,advanced=advanced,first_candidate=race['chosen'],selected_arm=selected,scores=scores))
    pd.DataFrame(raced,columns=['fold','arm','appearances','unique_patients']+METRICS).to_csv(PUB/'RACED_PATH_RESULTS.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='scores'} for r in choices]).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,actual_neural_fits=len(checks),logical_cells=len(expected),patient_metric_rows=len(frame),
        maximum_replay_error=maximum,all9_patient_metrics_independent=True,all_races_recomputed=True,random_initial_states_reproduced=True,
        ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_FRESH_ENERGY_INTERNAL',actual_neural_fits=len(checks),epochs=status['epochs'],logical_cells=len(expected),
        advanced_folds=[r['fold'] for r in choices if r['advanced']],final_stage_warranted=any(r['advanced'] for r in choices),new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('FRESH_RAW_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
