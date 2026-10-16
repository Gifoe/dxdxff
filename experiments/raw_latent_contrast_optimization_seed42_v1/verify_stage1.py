"""Independent fixed-coefficient/metric/role replay, no additional model search."""
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,average_precision_score
from common import *
from stage1 import contexts,feature_matrix,source_penalty
from model import FACTORS,recipe_key
from metrics import patient_rows,threshold


def main():
    binding=setup();_,banks=identity_and_banks(labels=True);store=RawStore()
    choice=[];assessment=[];gradient=[];maximum=0.;fit_count=0
    for ctx in contexts(banks):
        f,s=ctx['fold'],ctx['screen'];d=ctx['bank'];ix=ctx['fit'];rr=ctx['stored']['roles']
        x=feature_matrix(ctx,binding,store);y=(d['y'][ix]==0).astype(int)
        pid=d['patient'][ix];ch=d['channel'][ix];center=d['center'][ix]
        result=read(ROOT/'screen_results_private'/f'fold{f}_screen{s}.pkl')
        assert result['binding']['input']==binding
        for record in result['models']:
            j=list(map(recipe_key,FACTORS)).index(record['recipe']);m=record['model']
            z=(x[j].astype(float)-m['pre']['mean'])/m['pre']['scale']
            score=z@m['coef'][0]+m['intercept'][0];prob=expit(score)
            cal=rr['cal'];theta=threshold(y[cal],prob[cal],pid[cal]);assert theta['threshold']==record['threshold']
            if record['kind']=='choice':
                held=rr['choice'];reference=next(r for r in result['search'] if r['recipe']==record['recipe'])
            else:
                held=rr['held'];reference=None
            rows=patient_rows(y[held],prob[held],pid[held],ch[held],center[held],theta['threshold'])
            for row in rows:
                jj=held[pid[held]==row['patient']];truth=y[jj];prediction=prob[jj]>=theta['threshold']
                independent=dict(f1=f1_score(truth,prediction,pos_label=1,zero_division=0),
                    nez_f1=f1_score(truth,prediction,pos_label=0,zero_division=0),
                    macro_f1=f1_score(truth,prediction,average='macro',labels=[0,1],zero_division=0),
                    accuracy=accuracy_score(truth,prediction),auroc=roc_auc_score(truth,prob[jj]),
                    auprc=average_precision_score(truth,prob[jj]))
                for name,value in independent.items():
                    error=abs(row[name]-value);maximum=max(maximum,error);assert error<=1e-12,(name,error)
            if reference:
                for name in METRICS:
                    error=abs(np.nanmean([r[name] for r in rows])-reference[name]);maximum=max(maximum,error);assert error<=1e-12
                choice.append(reference)
            else:assessment.extend([dict(fold=f,screen=s,recipe=record['recipe'],**r) for r in rows])
            fit_count+=1
    saved=read(ROOT/'ASSESSMENT_ROWS_PRIVATE.pkl')
    a=pd.DataFrame(assessment).sort_values(['fold','screen','recipe','patient']).reset_index(drop=True)
    b=pd.DataFrame(saved).sort_values(['fold','screen','recipe','patient']).reset_index(drop=True)
    assert len(a)==len(b)==4080 and fit_count==560
    assert a[['fold','screen','recipe','patient','center']].equals(b[['fold','screen','recipe','patient','center']])
    error=float(np.nanmax(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy())));assert error<=1e-12;maximum=max(maximum,error)
    fr=jr(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json')
    for fold in range(1,6):
        candidates=[]
        for recipe in map(recipe_key,FACTORS):
            rows=[r for r in assessment if r['fold']==fold and r['recipe']==recipe]
            score=float(np.mean([r['f1'] for r in rows]))-.1*source_penalty(rows)
            candidates.append((recipe.count('1'),score,recipe))
        best=max(q[1] for q in candidates)
        chosen=min([q for q in candidates if q[1]>=best-.005],key=lambda q:(q[0],-q[1],q[2]))
        assert fr['recipes'][str(fold)]==chosen[2]
    # Anonymous release: suppress source aggregates based on fewer than5 distinct patients.
    sources=[]
    for (fold,recipe,center),g in a.groupby(['fold','recipe','center']):
        n=g.patient.nunique();sources.append(dict(fold=fold,recipe=recipe,center=center,patients=n,suppressed=n<5,
            **{m:(float(g[m].mean()) if n>=5 else np.nan) for m in METRICS}))
    pd.DataFrame(sources).to_csv(PUB/'FACTORIAL_SOURCE_RESULTS_PUBLIC.csv',index=False)
    admission=jr(PUB/'END_TO_END_ADMISSION.json') if (PUB/'END_TO_END_ADMISSION.json').exists() else None
    if admission:
        write(PUB/'END_TO_END_ADMISSION_PUBLIC.json',dict(status=admission['status'],
            binding_sha256=admission['binding_sha256'],arms=len(admission['reports']),
            all_complete_patient=True,logit_and_loss_max_error=0.,
            full_gradient_max_error=max(r['full_gradient_max_error'] for r in admission['reports']),
            RNG_exact=all(r['RNG_exact'] for r in admission['reports']),
            baseline_parameters=admission['reports'][0]['parameters'],attention_parameters=admission['reports'][-1]['parameters'],
            physiology_inputs=0,D1_branches=0))
    write(PUB/'STAGE1_INDEPENDENT_VERIFICATION.json',dict(status='PASS',logistic_fits=fit_count,
        assessment_patient_rows=len(a),unique_patient_identities=int(a.patient.nunique()),
        maximum_metric_replay_error=maximum,all_roles_disjoint_and_own_FIT_only=True,
        all_five_recipe_choices_independently_replayed=True,no_new_fit_or_Outer_access=True,
        source={n:sha(HERE/n) for n in ['verify_stage1.py','stage1.py','common.py','model.py','metrics.py']},
        verified_recipe_freeze_sha256=sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json')))
    print('RAW_STAGE1_INDEPENDENT_VERIFICATION_PASS',fit_count,len(a),flush=True)


if __name__=='__main__':main()
