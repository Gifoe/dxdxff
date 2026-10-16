"""Prospective candidate selector and complete-case paired bootstrap algebra."""
import numpy as np
METRICS=['accuracy','auroc','auprc','macro_f1','f1','nez_f1','mrr','hit_at_1','ez_fraction_bias']
STRICT=METRICS[:8]
def check_target(candidate,reference):
    rows=[dict(metric=m,candidate=float(candidate[m]),original_M8=float(reference[m]),delta=float(candidate[m]-reference[m]),passes=bool(candidate[m]>=reference[m])) for m in STRICT]
    return rows,all(r['passes'] for r in rows),all(candidate[m]>=reference[m] for m in ['auroc','auprc','macro_f1','f1'])
def choose(patients,protocol):
    assert set(patients.scope)=={'development'}
    rule=protocol['candidate_selection'];rows=[]
    for arm in rule['candidates']:
        g=patients[patients.model==arm]
        assert len(g)==65 and g.patient.nunique()==47
        weighted=sum(float(w)*float(g[m].mean()) for m,w in rule['weights'].items())
        folds=np.array([sum(float(w)*float(g[g.fold==f][m].mean()) for m,w in rule['weights'].items()) for f in range(1,6)])
        sd=float(folds.std(ddof=1));params=10377 if arm in ['S3','S4'] else 9329
        rows.append(dict(model=arm,weighted_mean=weighted,fold_score_sample_sd=sd,score=weighted-rule['fold_sampleSD_penalty']*sd,
            parameters=params,physiology_groups=6 if arm in ['S2','S4'] else 7,**{m:float(g[m].mean()) for m in METRICS}))
    best=max(r['score'] for r in rows);close=[r for r in rows if r['score']>=best-rule['close_score_allowance']]
    close.sort(key=lambda r:(r['parameters'],r['physiology_groups'],-r['score'],-r['macro_f1'],-r['auprc'],-r['auroc'],r['model']))
    return close[0]['model'],rows

def counts(ids):
    unique=sorted(set(ids));draw=np.random.default_rng(42).integers(0,len(unique),(10000,len(unique)))
    return unique,np.stack([(draw==i).sum(1) for i in range(len(unique))],axis=1),draw

def paired(a,b,ids,unique,multiplicity):
    valid=np.isfinite(a)&np.isfinite(b);delta=np.where(valid,a-b,np.nan)
    num=np.array([np.nansum(delta[np.asarray(ids)==p],axis=0) for p in unique])
    den=np.array([np.isfinite(delta[np.asarray(ids)==p]).sum(0) for p in unique])
    dd=multiplicity@den;dist=np.divide(multiplicity@num,dd,out=np.full(dd.shape,np.nan),where=dd>0)
    mean=np.divide(np.nansum(delta,axis=0),valid.sum(0),out=np.full(delta.shape[1],np.nan),where=valid.sum(0)>0)
    return mean,np.nanpercentile(dist,[2.5,97.5],axis=0),valid.sum(0),dist
