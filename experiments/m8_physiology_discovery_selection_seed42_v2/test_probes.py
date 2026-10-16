import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score,f1_score
from probes import metric_rows,threshold,preprocess


def test_all_metrics_including_probability_ties():
    rng=np.random.default_rng(42)
    pid=np.repeat(np.array(['A','B','C']),31);ch=np.array([f'C{i:02}' for i in range(31)]*3)
    y=rng.integers(0,2,len(pid));score=np.round(rng.uniform(size=len(pid)),1)
    rr=metric_rows(y,score,pid,ch,pid,.4)
    for r in rr:
        take=pid==r['patient'];a=y[take];p=score[take];v=p>=.4
        for key,value in dict(accuracy=accuracy_score(a,v),auroc=roc_auc_score(a,p),auprc=average_precision_score(a,p),macro_f1=f1_score(a,v,labels=[0,1],average='macro'),f1=f1_score(a,v),nez_f1=f1_score(1-a,1-v.astype(int))).items():
            assert abs(r[key]-value)<1e-12
        rank=a[np.lexsort((ch[take],-p))]
        assert r['mrr']==1/(np.flatnonzero(rank)[0]+1) and r['hit_at_1']==rank[0]


def test_threshold_only_uses_calibration_labels():
    rng=np.random.default_rng(13);pid=np.repeat(np.array(['A','B','C']),19)
    y=rng.integers(0,2,len(pid));score=rng.uniform(size=len(pid));ch=np.tile(np.arange(19).astype(str),3)
    candidates=[]
    for t in np.round(np.arange(0,1.0001,.005),3):
        rows=metric_rows(y,score,pid,ch,pid,t)
        candidates.append((np.mean([r['macro_f1'] for r in rows]),np.mean([r['f1'] for r in rows]),-abs(t-.5),-t,t))
    assert threshold(y,score,pid)==max(candidates)[-1]


def test_heldout_values_do_not_fit_imputer_or_scaler():
    a=np.array([[1.,np.nan],[3.,4.],[9.,8.]])
    x,p=preprocess(a,np.array([0,1]));b=a.copy();b[2]=12345
    xx,pp=preprocess(b,np.array([0,1]))
    for name in p:np.testing.assert_array_equal(p[name],pp[name])
    np.testing.assert_array_equal(x[:2],xx[:2])
