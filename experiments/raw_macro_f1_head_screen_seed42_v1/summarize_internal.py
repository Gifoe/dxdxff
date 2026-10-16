"""Replay readout states, verify patient metrics, then freeze fold-local selection."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from head import Head,ARMS,RECIPES
from train_internal import predict,select


def main():
    binding=setup();status=jr(PUB/'INTERNAL_STATUS.json');assert status['status']=='ALL120_INTERNAL_HEAD_CELLS_COMPLETE'
    seal=jr(ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json');assert seal['input']==binding and sha(ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json')==status['seal_sha256']
    rows=[];checks=[];maximum=0.
    for ctx in contexts():
        z,cp=inputs(ctx);d=ctx['data'];rr=ctx['roles']
        for arm in ARMS:
            cell=ROOT/'heads_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm
            result=read(cell/'SCORES_PRIVATE.pkl');q=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            source=next(r for r in seal['cells'] if r['fold']==ctx['fold'] and r['screen']==ctx['screen'] and r['arm']==arm)
            assert sha(cell/'SCORES_PRIVATE.pkl')==source['sha256'] and sha(cell/'BEST_PRIVATE.pt')==result['best_sha256']
            model=Head(arm,cp['model'],ctx['fold']);model.load_state_dict(q['model'])
            replay=predict(model,z[RECIPES[arm]],d['patient']);error=float(np.max(np.abs(replay-result['logit_nez'])))
            assert error<=1e-6;maximum=max(maximum,error)
            assert select(d,rr['cal'],result['logit_nez'][rr['cal']])==q['selection']
            held=rr['held'];part=patient_rows(d,held,result['logit_nez'][held],result['threshold'])
            for r in part:
                loc=d['patient'][held]==r['patient'];y=1-d['y'][held][loc]
                pn=probability(result['logit_nez'][held][loc]);prediction=pn<result['threshold'];p=1-pn.astype(float)
                independent=dict(macro_f1=f1_score(y,prediction,average='macro',zero_division=0),
                    f1=f1_score(y,prediction,pos_label=1,zero_division=0),nez_f1=f1_score(y,prediction,pos_label=0,zero_division=0),
                    accuracy=accuracy_score(y,prediction),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p))
                for m,v in independent.items():assert abs(r[m]-v)<1e-12
            rows.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**r) for r in part])
            checks.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,logit_max_error=error))
        print('MACRO_HEAD_PATH_VERIFIED',ctx['fold'],ctx['screen'],flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl'))
    keys=['fold','screen','arm','patient'];frame=frame.sort_values(keys).reset_index(drop=True);old=old.sort_values(keys).reset_index(drop=True)
    assert frame[keys].equals(old[keys]) and np.max(np.abs(frame[METRICS].to_numpy()-old[METRICS].to_numpy()))<1e-12
    frame.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'INTERNAL_MAIN_RESULTS.csv',index=False)
    frame.groupby(['arm','fold'])[METRICS].mean().reset_index().to_csv(PUB/'INTERNAL_FOLD_RESULTS.csv',index=False)
    sources=[]
    for (arm,center),g in frame.groupby(['arm','center']):
        n=g.patient.nunique();sources.append(dict(arm=arm,center=center,unique_patients=n,appearances=len(g),suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    pd.DataFrame(sources).to_csv(PUB/'INTERNAL_SOURCE_RESULTS.csv',index=False)
    absolute=[];paired=[]
    for arm,g in frame.groupby('arm'):absolute.extend([dict(arm=arm,**r) for r in cluster_interval(g.to_dict('records'))])
    comparisons=[(a,'REFIT') for a in ARMS if a!='REFIT']+[('REFIT','NATIVE'),('PATIENT','LOCAL')]
    for left,right in comparisons:
        a=frame[frame.arm==left].merge(frame[frame.arm==right],on=['fold','screen','patient','center'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=a.patient,**{m:a[m+'_l']-a[m+'_r'] for m in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(absolute).to_csv(PUB/'INTERNAL_ABSOLUTE_INTERVALS.csv',index=False)
    pd.DataFrame(paired).to_csv(PUB/'INTERNAL_PAIRED_INTERVALS.csv',index=False)
    choices=[]
    for fold in range(1,6):
        results=[]
        for arm in ARMS:
            g=frame[(frame.fold==fold)&(frame.arm==arm)];s=g.groupby('center').macro_f1.mean()
            penalized=float(g.macro_f1.mean())-.1*(float(s.std(ddof=1)) if len(s)>1 else 0.)
            params=0 if arm=='NATIVE' else (14097 if arm in ['PATIENT','LOCAL'] else 10369)
            results.append(dict(arm=arm,score=penalized,parameters=params,macro_f1=float(g.macro_f1.mean())))
        highest=max(r['score'] for r in results)
        chosen=min([r for r in results if r['score']>=highest-.005],key=lambda r:(r['parameters'],-r['score'],ARMS.index(r['arm'])))
        choices.append(dict(fold=fold,chosen=chosen['arm'],score=chosen['score'],internal_macro_f1=chosen['macro_f1'],all_scores=results))
    freeze(ROOT/'OWN_FIT_HEAD_SELECTION_PRIVATE.json',dict(input=binding,seal=sha(ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json'),choices=choices))
    pd.DataFrame([{k:v for k,v in r.items() if k!='all_scores'} for r in choices]).to_csv(PUB/'OWN_FIT_HEAD_SELECTION.csv',index=False)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',input_binding_sha256=digest(binding),logical_cells=len(checks),
        patient_rows=len(frame),appearances_per_arm=len(frame[frame.arm=='NATIVE']),unique_held_patients=frame.patient.nunique(),
        maximum_score_replay_error=maximum,all_metrics_recomputed=True,selection_ownFIT_only=True,
        no_original_Development_or_Outer_used=True,selection_sha256=sha(ROOT/'OWN_FIT_HEAD_SELECTION_PRIVATE.json'),checks=checks))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_MACRO_HEAD_INTERNAL',
        logical_cells=120,actual_head_fits=105,epochs=status['epochs'],new_Outer_passes=0,
        goal_established=False,scope='internal selection, not final test or independent confirmation'))
    print('MACRO_HEAD_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
