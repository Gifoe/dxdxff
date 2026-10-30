"""Independent reconstruction from the high-gain anonymous channel export."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
METRICS=['auroc','macro_f1','top1','recall_at_10pct','recall_at_20pct']
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check(public):
    public=Path(public);channels=rows(public/'CHANNEL_SCORES_ANONYMOUS.csv');reported=rows(public/'CASE_METRICS.csv');curves=rows(public/'RECALL_AT_K.csv');meta={r['case_id']:r for r in rows(public/'CASE_METADATA.csv')}
    keys=sorted({(r['case_id'],r['model'],r['seed'],r['condition']) for r in channels});assert len(keys)==8
    computed={};curve_lookup={(r['case_id'],r['model'],r['seed'],r['condition'],int(r['k'])):r for r in curves}
    for key in keys:
        group=[r for r in channels if tuple(r[k] for k in ['case_id','model','seed','condition'])==key]
        y=np.array([int(r['truth_ez']) for r in group]);score=np.array([float(r['score_ez']) for r in group]);pred=np.array([int(r['prediction_ez']) for r in group])
        n=len(y);assert n==int(meta[key[0]]['total_channels']) and int(y.sum())==int(meta[key[0]]['ez_channels'])
        order=sorted(range(n),key=lambda i:(-score[i],group[i]['channel_id']))
        assert [int(group[i]['rank_ez']) for i in order]==list(range(1,n+1))
        assert all(r['truth_label']==('EZ' if int(r['truth_ez']) else 'NEZ') for r in group)
        rr=dict(auroc=float(roc_auc_score(y,score)),macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro')),top1=float(y[order[0]]))
        for q in [.1,.2]:rr[f'recall_at_{round(q*100)}pct']=float(y[order[:math.ceil(q*n)]].sum()/y.sum())
        computed[key]=rr
        for k in range(n+1):
            r=curve_lookup[key+(k,)];hits=sum(int(y[i]) for i in order[:k])
            assert int(r['ez_recovered'])==hits and int(r['ez_total'])==y.sum()
            assert abs(float(r['recall'])-hits/y.sum())<1e-12 and abs(float(r['review_fraction'])-k/n)<1e-12
            if k:assert abs(float(r['precision'])-hits/k)<1e-12
            else:assert r['precision']==''
    for case in meta:
        for cond in ['NO_PR','PR']:
            computed[case,'DRECHANNET','MEAN_42_43_44',cond]={m:float(np.mean([computed[case,'DRECHANNET',str(s),cond][m] for s in [42,43,44]])) for m in METRICS}
    assert len(reported)==25
    for r in reported:
        no,pr=[computed[r['case_id'],r['model'],r['seed'],c][r['metric']] for c in ['NO_PR','PR']]
        assert max(abs(no-float(r['no_pr'])),abs(pr-float(r['pr'])),abs(pr-no-float(r['change'])))<1e-12
        assert 0<=float(r['percentile_among_all80'])<=100 and 0<=float(r['percentile_among_case_pool'])<=100
    for case in meta:
        assert case=='CASE-H01';sign=1
        changes=[{m:computed[case,'DRECHANNET',str(s),'PR'][m]-computed[case,'DRECHANNET',str(s),'NO_PR'][m] for m in METRICS} for s in [42,43,44]]
        assert sum(all(sign*r[m]>1e-12 for m in ['auroc','recall_at_10pct']) for r in changes)==int(meta[case]['dre_joint_consistent_seeds'])
        assert int(meta[case]['total_channels'])>=20 and int(meta[case]['ez_channels'])>0
    print('ANONYMOUS_SELECTED_CHANNELS_ALL25_METRIC_ROWS_ALL_K_CURVES_INDEPENDENT_PASS',len(channels),len(curves))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('public',type=Path);check(p.parse_args().public)
