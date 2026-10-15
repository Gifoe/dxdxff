"""Equivalent canonical ranking without native Unicode np.lexsort dispatch."""
import numpy as np


def ranking(y,p,channel):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);channel=[str(c) for c in channel]
    assert len(y)==len(p)==len(channel) and np.isfinite(p).all()
    names=['mrr','hit_at_1','hit_at_3','hit_at_5','precision_at_3','recall_at_3','ndcg_at_5']
    positive=int(y.sum())
    if not positive:return dict.fromkeys(names,np.nan)
    order=sorted(range(len(y)),key=lambda i:(-float(p[i]),channel[i]))
    yy=y[np.asarray(order,dtype=np.int64)];first=next(i for i,v in enumerate(yy) if v)
    k3=min(3,len(y));k5=min(5,len(y));gains=1/np.log2(np.arange(k5)+2)
    return dict(mrr=1/(first+1),hit_at_1=float(bool(yy[0])),hit_at_3=float(bool(yy[:k3].any())),hit_at_5=float(bool(yy[:k5].any())),
        precision_at_3=float(yy[:k3].sum()/k3),recall_at_3=float(yy[:k3].sum()/positive),
        ndcg_at_5=float((yy[:k5]*gains).sum())/float(gains[:min(positive,k5)].sum()))
