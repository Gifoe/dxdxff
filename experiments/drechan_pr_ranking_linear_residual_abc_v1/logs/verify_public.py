"""Reconstruct all published source means, ranks and budgets independently."""
import csv,argparse,math
from pathlib import Path
def read(p,n):
    with (p/(n+'.csv')).open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    for r in rr:
        if r.get('model')=='TIMECONV':
            if r.get('condition')=='PR_LINEAR_RESIDUAL':r['condition']='FULL_PR'
            if r.get('contrast','').startswith('PR_LINEAR_RESIDUAL_MINUS_'):r['contrast']=r['contrast'].replace('PR_LINEAR_RESIDUAL_MINUS_','FULL_MINUS_')
    return rr

def check(p):
    a={(r['model'],r['condition'],r['metric']):r for r in read(p,'TABLE_A')};s=read(p,'TABLE_C_SOURCE');cm={(r['model'],r['condition']):r for r in read(p,'TABLE_C_SUMMARY')}
    for (model,arm,metric),r in a.items():
        rr=[x for x in s if (x['model'],x['condition'])==(model,arm)];assert sorted(int(x['patients']) for x in rr)==[8,15,21,36]
        value=sum(int(x['patients'])*float(x[metric]) for x in rr)/80;assert abs(value-float(r['mean']))<1e-12
        assert abs(sum(float(x[metric]) for x in rr)/4-float(cm[model,arm]['center_mean_'+metric]))<1e-12
        if metric!='ez_fraction_bias':assert abs(min(float(x[metric]) for x in rr)-float(cm[model,arm]['worst_'+metric]))<1e-12
        if arm=='SCORE_Z' and metric in ['auroc','auprc','mrr','hit_at_1','hit_at_3','hit_at_5']:assert float(r['mean'])==float(a[model,'NO_PR',metric]['mean'])
    for r in read(p,'TABLE_B'):
        other=r['contrast'].replace('FULL_MINUS_','');d=float(a[r['model'],'FULL_PR',r['metric']]['mean'])-float(a[r['model'],other,r['metric']]['mean'])
        assert abs(d-float(r['mean']))<1e-12
        assert abs(sum(float(r[k]) for k in ['positive_fraction','zero_fraction','negative_fraction'])-1)<1e-12
        if int(r['family_size']):assert float(r['adjusted_low'])<=float(r['low'])+1e-12 and float(r['adjusted_high'])>=float(r['high'])-1e-12
    budget={(r['model'],r['condition'],r['budget_kind'],float(r['budget']),r['metric']):r for r in read(p,'TABLE_D_REVIEW')}
    curves=read(p,'COVERAGE_CURVE')
    for model in ['LINEAR_SVM','TIMECONV','DRECHANNET']:
        for arm in ['NO_PR','FULL_PR']:
            rr=sorted([r for r in curves if (r['model'],r['condition'])==(model,arm)],key=lambda r:float(r['review_fraction']))
            assert len(rr)==101 and abs(float(rr[0]['mean']))<1e-12 and abs(float(rr[-1]['mean'])-1)<1e-12
            assert all(float(y['mean'])>=float(x['mean'])-1e-12 for x,y in zip(rr[:-1],rr[1:]))
            for q in [.05,.1,.2]:
                r=next(r for r in rr if float(r['review_fraction'])==q);assert abs(float(r['mean'])-float(budget[model,arm,'fraction',q,'recall']['mean']))<1e-12
    for r in read(p,'TABLE_D_PAIRED'):
        d=float(budget[r['model'],'FULL_PR',r['budget_kind'],float(r['budget']),r['metric']]['mean'])-float(budget[r['model'],'NO_PR',r['budget_kind'],float(r['budget']),r['metric']]['mean'])
        assert abs(d-float(r['mean']))<1e-12
    if (p/'TABLE_E.csv').exists():
        e={(r['model'],r['condition'],r['K'],r['metric']):r for r in read(p,'TABLE_E')}
        for r in read(p,'TABLE_E_PAIRED'):
            label=r['contrast'];model=r['model'];metric=r['metric']
            if label.startswith('FULL_MINUS_NOPR_'):
                k=label[-1];x,y=e[model,'FULL_PR',k,metric],e[model,'NO_PR',k,metric]
            else:
                arm,k=label.split('_K');k=k[0];x,y=e[model,arm,k,metric],e[model,arm,'All',metric]
            assert abs(float(r['mean'])-(float(x['mean'])-float(y['mean'])))<1e-12
    print('LOCAL_PUBLIC_SOURCE_RANK_BUDGET_AND_PAIRED_CHECK_PASS',len(a))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('public',type=Path);check(p.parse_args().public)
