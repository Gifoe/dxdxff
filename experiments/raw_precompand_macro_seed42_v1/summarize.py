"""Independent full-state CAL/HELD replay and ownFIT-only selection."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model,ARMS,TRAINED
from train import load_context,predict,native


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL30_MATRIX_FITS_COMPLETE'
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding and status['seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    rows=[];checks=[];maximum=0.
    for ctx in contexts():
        d=ctx['data'];rr=ctx['roles'];cal=rr['cal'];held=rr['held'];caches=load_context(ctx,binding)
        _,part=native(ctx);rows.extend(part)
        for arm in TRAINED:
            cell=ROOT/'matrices_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm
            hp=cell/'HELD_PRIVATE.pkl';bp=cell/'BEST_PRIVATE.pt'
            source=next(r for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(ctx['fold'],ctx['screen'],arm))
            assert sha(hp)==source['sha256'];q=read(hp);state=torch.load(bp,map_location='cpu',weights_only=False)
            complete=jr(cell/'COMPLETE_PRIVATE.json')
            assert complete['best_sha256']==sha(bp) and complete['held_sha256']==sha(hp)
            assert state['binding']==q['binding'] and q['binding']['input']==binding
            assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
            model=Model(arm,ctx['fold'],ctx['reference_state']).cuda()
            with torch.no_grad():model.matrix.copy_(state['matrix'])
            cl=predict(model,ctx,cal,caches);hpredict=predict(model,ctx,held,caches)
            error=max(float(np.max(np.abs(cl-q['cal_logit_nez']))),float(np.max(np.abs(hpredict-q['logit_nez']))))
            assert error<=1e-6;maximum=max(maximum,error)
            assert calibrate(d,cal,cl)==state['selection']
            part=held_rows(ctx,hpredict,state['selection']['threshold'],arm)
            for r in part:
                ix=d['patient'][held]==r['patient'];yy=1-d['y'][held][ix];pn=probability(hpredict[ix]);pp=1-pn.astype(float);pred=pn<state['selection']['threshold']
                independent=dict(macro_f1=f1_score(yy,pred,average='macro',zero_division=0),f1=f1_score(yy,pred,pos_label=1,zero_division=0),
                                 nez_f1=f1_score(yy,pred,pos_label=0,zero_division=0),accuracy=accuracy_score(yy,pred),
                                 auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp))
                order=np.lexsort((d['channel'][held][ix],-pp));first=int(np.flatnonzero(yy[order])[0])
                independent.update(mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-yy.mean()))
                for m,v in independent.items():assert abs(v-r[m])<1e-12
            rows.extend(part);checks.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,maximum_replay_error=error))
        del caches;gc.collect();torch.cuda.empty_cache()
        print('PRECOMPAND_PATH_INDEPENDENTLY_VERIFIED',ctx['fold'],ctx['screen'],flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl'));keys=['fold','screen','arm','patient']
    a=frame.sort_values(keys).reset_index(drop=True);b=old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    frame.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'INTERNAL_MAIN_RESULTS.csv',index=False)
    frame.groupby(['arm','fold'])[METRICS].mean().reset_index().to_csv(PUB/'INTERNAL_FOLD_RESULTS.csv',index=False)
    sources=[];absolute=[];paired=[];choices=[]
    for (arm,center),g in frame.groupby(['arm','center']):
        n=g.patient.nunique();sources.append(dict(arm=arm,center=center,appearances=len(g),unique_patients=n,suppressed=n<5,
                         **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for arm,g in frame.groupby('arm'):
        absolute.extend([dict(arm=arm,**r) for r in parent.cluster_interval(g.to_dict('records'))])
    for left,right in [('WINDOW_RESIDUAL','NATIVE'),('PRECOMPAND_RESIDUAL','NATIVE'),('PRECOMPAND_RESIDUAL','WINDOW_RESIDUAL')]:
        m=frame[frame.arm==left].merge(frame[frame.arm==right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=m.patient,**{k:m[k+'_l']-m[k+'_r'] for k in METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in parent.cluster_interval(diff.to_dict('records'))])
    for fold in range(1,6):
        results=[]
        for arm in ARMS:
            g=frame[(frame.fold==fold)&(frame.arm==arm)];s=g.groupby('center').macro_f1.mean()
            score=float(g.macro_f1.mean())-.1*(float(s.std(ddof=1)) if len(s)>1 else 0.)
            results.append(dict(arm=arm,score=score,parameters=0 if arm=='NATIVE' else 576))
        high=max(r['score'] for r in results)
        chosen=min([r for r in results if r['score']>=high-.005],key=lambda r:(r['parameters'],-r['score'],ARMS.index(r['arm'])))
        choices.append(dict(fold=fold,selected_arm=chosen['arm'],score=chosen['score'],all_scores=results))
    pd.DataFrame(sources).to_csv(PUB/'INTERNAL_SOURCE_RESULTS.csv',index=False)
    pd.DataFrame(absolute).to_csv(PUB/'INTERNAL_ABSOLUTE_CI.csv',index=False)
    pd.DataFrame(paired).to_csv(PUB/'INTERNAL_PAIRED_CI.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='all_scores'} for r in choices]).to_csv(PUB/'OWN_FIT_SELECTION.csv',index=False)
    freeze(ROOT/'OWN_FIT_SELECTION_PRIVATE.json',dict(input=binding,internal_seal=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),choices=choices))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checks=checks,matrix_fits=len(checks),patient_metric_rows=len(frame),
          maximum_replay_error=maximum,independent_sklearn_metrics_pass=True,ownFIT_selection_only=True,
          ownFIT_selection_sha256=sha(ROOT/'OWN_FIT_SELECTION_PRIVATE.json'),original_Development_Outer_used=False))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_PRECOMPAND_INTERNAL',matrix_fits=30,logical_cells=45,
          epochs=status['epochs'],new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('PRECOMPAND_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
