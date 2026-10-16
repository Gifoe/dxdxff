"""Reconstruct every executed internal score,then anonymous selection/ablation summaries."""
import gc
from scipy.special import expit
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from model import Model,ARMS,CANDIDATES,ABLATIONS
from train_internal import evaluate,calibration


def cluster_interval(rows,columns=METRICS):
    frame=pd.DataFrame(rows);ids=sorted(frame.patient.unique());sums=[];counts=[]
    for pid in ids:
        g=frame[frame.patient==pid];sums.append(g[columns].sum().to_numpy(float));counts.append(len(g))
    sums=np.stack(sums);counts=np.asarray(counts)
    draw=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)))
    sample=sums[draw].sum(1)/counts[draw].sum(1)[:,None]
    result=[]
    for j,m in enumerate(columns):
        lo,hi=np.quantile(sample[:,j],[.025,.975])
        result.append(dict(metric=m,mean=float(frame[m].mean()),low=float(lo),high=float(hi),
            logical_patient_appearances=len(frame),unique_patients=len(ids),draws=10000,
            interval='descriptive_selection_conditioned_patient_ID_cluster_percentile'))
    return result


def main():
    binding=setup();status=jr(PUB/'RUN_STATUS.json');assert status['status']=='ALL_REGISTERED_INTERNAL_CELLS_COMPLETE'
    assert status['seal_sha256']==sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    seal=jr(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');assert seal['input']==binding
    banks=legal_fit_banks(labels=True);paths=role_paths(banks);store=RawStore();rows=[];checks=[];maximum=0.
    for record in seal['cells']:
        fold,screen,arm=record['fold'],record['screen'],record['arm'];d=banks[fold-1];rr=paths[(fold,screen)]['roles']
        cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;cp=cell/'BEST_PRIVATE.pt';hp=cell/'HELD_PREDICTIONS_PRIVATE.pkl'
        assert sha(cp)==record['checkpoint_sha256'] and sha(hp)==record['held_predictions_sha256']
        saved=read(hp);checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
        model=Model(arm,fold).cuda();model.load_state_dict(checkpoint['model'])
        cal=evaluate(model,d,rr['cal'],store);selected=calibration(d,rr['cal'],cal)
        assert selected==checkpoint['selection'] and selected['threshold']==saved['threshold']
        error=float(np.max(np.abs(cal-saved['cal_logit_nez'])));assert error<=1e-6;maximum=max(maximum,error)
        held=rr['held'];assert np.array_equal(d['archive_indices'][held],saved['archive_indices'])
        p=expit(-saved['logit_nez'].astype(float));truth=1-d['y'][held]
        result=metrics.patient_rows(truth,p,d['patient'][held],d['channel'][held],d['center'][held],saved['threshold'])
        for r in result:
            loc=d['patient'][held]==r['patient'];y=truth[loc];pp=p[loc];prediction=pp>=saved['threshold']
            reference=dict(f1=f1_score(y,prediction,pos_label=1,zero_division=0),nez_f1=f1_score(y,prediction,pos_label=0,zero_division=0),
                macro_f1=f1_score(y,prediction,average='macro',zero_division=0),accuracy=accuracy_score(y,prediction),
                auroc=roc_auc_score(y,pp),auprc=average_precision_score(y,pp))
            for name,v in reference.items():assert abs(r[name]-v)<=1e-12
            order=sorted(range(int(loc.sum())),key=lambda k:(-pp[k],str(d['channel'][held][loc][k])))
            ranks=[i+1 for i,k in enumerate(order) if y[k]]
            assert r['mrr']==1/ranks[0] and r['hit_at_1']==float(y[order[0]])
        rows.extend([dict(fold=fold,screen=screen,arm=arm,**r) for r in result])
        checks.append(dict(fold=fold,screen=screen,arm=arm,cal_logit_max_error=error,selected_epoch=checkpoint['epoch']))
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
        print('RETENTION_INDEPENDENT_CELL_VERIFIED',len(checks),'of',len(seal['cells']),flush=True)
    frame=pd.DataFrame(rows);old=pd.DataFrame(read(ROOT/'PATIENT_RESULTS_PRIVATE.pkl'))
    sort=['fold','screen','arm','patient'];frame=frame.sort_values(sort).reset_index(drop=True);old=old.sort_values(sort).reset_index(drop=True)
    assert frame[sort].equals(old[sort]) and np.nanmax(np.abs(frame[METRICS].to_numpy()-old[METRICS].to_numpy()))<=1e-12
    first=frame[frame.screen==1];primary=first.groupby('arm')[METRICS].mean().reset_index()
    primary.to_csv(PUB/'FIRST_PATH_MAIN_RESULTS.csv',index=False)
    first.groupby(['arm','fold'])[METRICS].mean().reset_index().to_csv(PUB/'FIRST_PATH_FOLD_RESULTS.csv',index=False)
    source=[]
    for (arm,center),g in first.groupby(['arm','center']):
        n=g.patient.nunique();source.append(dict(arm=arm,center=center,unique_patients=n,logical_appearances=len(g),suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    pd.DataFrame(source).to_csv(PUB/'FIRST_PATH_SOURCE_RESULTS.csv',index=False)
    intervals=[];contrasts=[]
    for arm,g in first.groupby('arm'):
        intervals.extend([dict(arm=arm,**r) for r in cluster_interval(g.to_dict('records'))])
    base=first[first.arm=='C0_WINDOW_SHORT']
    comparisons=[(a,'C0_WINDOW_SHORT') for a in ARMS if a!='C0_WINDOW_SHORT' and a in set(first.arm)]
    comparisons.extend([(a,b) for a,b in [('C3_PRE_LONG','C2_WINDOW_LONG'),('C3_PRE_LONG','C1_PRE_SHORT'),('C4_PRE_LONG_CONTEXT','C3_PRE_LONG')] if a in set(first.arm) and b in set(first.arm)])
    for left,right in comparisons:
        joined=first[first.arm==left].merge(first[first.arm==right],on=['fold','screen','patient','center'],suffixes=('_l','_r'),validate='one_to_one')
        paired=pd.DataFrame(dict(patient=joined.patient,**{m:joined[m+'_l']-joined[m+'_r'] for m in METRICS}))
        contrasts.extend([dict(left=left,right=right,**r) for r in cluster_interval(paired.to_dict('records'))])
    pd.DataFrame(intervals).to_csv(PUB/'FIRST_PATH_ABSOLUTE_INTERVALS.csv',index=False)
    pd.DataFrame(contrasts).to_csv(PUB/'FIRST_PATH_PAIRED_INTERVALS.csv',index=False)
    races=seal['racing'];race_rows=[]
    for decision in races:
        if not decision['advanced']:continue
        f=decision['fold'];candidate=decision['chosen']
        for s in [2,3]:
            pair=frame[(frame.fold==f)&(frame.screen==s)]
            for arm in ['C0_WINDOW_SHORT',candidate]:
                g=pair[pair.arm==arm]
                assert len(g)>0
                race_rows.append(dict(fold=f,screen=s,arm=arm,patients=len(g),**{m:float(g[m].mean()) for m in METRICS}))
    pd.DataFrame(race_rows,columns=['fold','screen','arm','patients']+METRICS).to_csv(PUB/'ADDITIONAL_PATH_RESULTS.csv',index=False)
    write(PUB/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',input_binding_sha256=digest(binding),cells=len(checks),
        patient_rows=len(frame),calibration_replay_max_error=maximum,all_saved_score_metrics_recomputed=True,
        private_rows_not_transferred=True,original_Development_and_Outer_unused=True,
        protected_previous_study_unchanged=True,checks=checks,
        first_path_logical_patients_per_arm=int(first[first.arm=='C0_WINDOW_SHORT'].shape[0]),
        first_path_unique_patients=int(first.patient.nunique())))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY',
        executed_cells=len(checks),formal_epochs=status['epochs'],new_Outer_passes=0,
        previous_Outer_not_an_optimization_input=True,target_F1_still_requires_frozen_held_out_verification=True,
        no_independent_confirmation_claim=True))
    print('RETENTION_INTERNAL_VERIFIED_COMPLETE',len(checks),flush=True)


if __name__=='__main__':main()
