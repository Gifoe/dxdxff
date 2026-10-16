"""Independent frozen-state replay and ownFIT-only final configuration freeze."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model,ARMS,TRAINED
from admit import evaluate
from train_internal import native,penalized,choose


def summarize_group(frame,prefix):
    frame.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/(prefix+'_MAIN.csv'),index=False)
    frame.groupby(['fold','arm'])[METRICS].mean().reset_index().to_csv(PUB/(prefix+'_FOLD.csv'),index=False)
    source=[];absolute=[];paired=[]
    for (arm,center),g in frame.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
                         **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in frame.groupby('arm'):
        absolute.extend([dict(arm=arm,**r) for r in parent.cluster_interval(g.to_dict('records'))])
    pairs=[(a,'NATIVE') for a in TRAINED]+[(a,'MEAN_TUNE') for a in TRAINED[1:]]+[('DISPERSION','MEAN_RESIDUAL')]
    for left,right in pairs:
        a=frame[frame.arm==left].merge(frame[frame.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=a.patient,**{m:a[m+'_l']-a[m+'_r'] for m in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in parent.cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(source).to_csv(PUB/(prefix+'_SOURCE.csv'),index=False)
    pd.DataFrame(absolute).to_csv(PUB/(prefix+'_ABSOLUTE_CI.csv'),index=False)
    pd.DataFrame(paired).to_csv(PUB/(prefix+'_PAIRED_CI.csv'),index=False)


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_INTERNAL_CELLS_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding and sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')==status['seal_sha256']
    cc={(q['fold'],q['screen']):q for q in contexts()};store=RawStore();expected={(f,1,a) for f in range(1,6) for a in ARMS}
    for r in seal['races']:
        if r['advanced']:
            expected.update((r['fold'],s,a) for s in [2,3] for a in dict.fromkeys(['NATIVE','MEAN_TUNE',r['chosen']]))
    actual=pd.read_csv(PUB/'INTERNAL_CELL_RESULTS.csv')
    assert set(zip(actual.fold,actual.screen,actual.arm))==expected and len(actual)==len(expected)
    rows=[];checks=[];maximum=0.
    for fold,screen,arm in sorted(expected):
        ctx=cc[(fold,screen)];d=ctx['data'];rr=ctx['roles'];cal=rr['cal'];held=rr['held']
        if arm=='NATIVE':summary,part=native(ctx,binding);rows.extend(part);continue
        cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;hp=cell/'HELD_PRIVATE.pkl';bp=cell/'BEST_PRIVATE.pt'
        source=next(q for q in seal['cells'] if (q['fold'],q['screen'],q['arm'])==(fold,screen,arm))
        assert sha(hp)==source['sha256'];q=read(hp);best=torch.load(bp,map_location='cpu',weights_only=False)
        assert best['binding']==q['binding'] and q['binding']['input']==binding
        assert sha(bp)==q['summary']['checkpoint_sha256']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][held])
        assert np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
        model=Model(arm,fold,ctx['reference_state']).cuda();model.load_state_dict(best['model'])
        replay=evaluate(model,d,cal,store);held_replay=evaluate(model,d,held,store)
        error=max(float(np.max(np.abs(replay-q['cal_logit_nez']))),float(np.max(np.abs(held_replay-q['logit_nez']))))
        assert error<=1e-6;maximum=max(maximum,error)
        assert calibrate(d,cal,replay)==best['selection']
        part=held_rows(ctx,held_replay,best['selection']['threshold'],arm)
        for row in part:
            loc=d['patient'][held]==row['patient'];yy=1-d['y'][held][loc];pn=probability(held_replay[loc]);pr=pn<best['selection']['threshold'];pp=1-pn.astype(float)
            independent=dict(macro_f1=f1_score(yy,pr,average='macro',zero_division=0),f1=f1_score(yy,pr,pos_label=1,zero_division=0),
                             nez_f1=f1_score(yy,pr,pos_label=0,zero_division=0),auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),
                             accuracy=accuracy_score(yy,pr))
            for m,v in independent.items():assert abs(v-row[m])<1e-12
        rows.extend(part);checks.append(dict(fold=fold,screen=screen,arm=arm,maximum_logit_replay_error=error))
        del model;gc.collect();torch.cuda.empty_cache()
        print('DISPERSION_CELL_INDEPENDENTLY_VERIFIED',fold,screen,arm,flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a=frame.sort_values(keys).reset_index(drop=True);b=old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    summarize_group(frame[frame.screen==1],'FIRST_PATH')
    choices=[];raced=[]
    for race in seal['races']:
        fold=race['fold'];selected='NATIVE';score=None
        if race['advanced']:
            arms=list(dict.fromkeys(['NATIVE','MEAN_TUNE',race['chosen']]))
            results=[]
            for arm in arms:
                g=frame[(frame.fold==fold)&(frame.arm==arm)];assert set(g.screen)=={1,2,3}
                params=0 if arm=='NATIVE' else (12529 if arm=='MEAN_TUNE' else 13105)
                results.append(dict(arm=arm,parameters=params,penalized_macro=penalized(g.to_dict('records'))))
            best=choose(results);selected=best['arm'];score=best['penalized_macro']
            for arm in arms:
                g=frame[(frame.fold==fold)&(frame.arm==arm)]
                raced.append(dict(fold=fold,arm=arm,appearances=len(g),unique_patients=g.patient.nunique(),
                                  **{m:float(g[m].mean()) for m in METRICS}))
        choices.append(dict(fold=fold,advanced=race['advanced'],firstpath_candidate=race['chosen'],selected_arm=selected,
                            score=score,selection_ownFIT_only=True))
    pd.DataFrame(raced).to_csv(PUB/'RACED_PATH_RESULTS.csv',index=False)
    pd.DataFrame(choices).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,actual_trainable_fits=len(checks),
          logical_cells=len(expected),patient_metric_rows=len(frame),maximum_logit_replay_error=maximum,
          all_metrics_recomputed=True,score_seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
          ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_DISPERSION_INTERNAL',actual_trainable_fits=len(checks),
          epochs=status['epochs'],logical_cells=len(expected),new_Outer_passes=0,goal_established=False,
          interpretation='Previously used ownFIT screening data,not independent confirmation or full80patient final test.'))
    print('DISPERSION_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
