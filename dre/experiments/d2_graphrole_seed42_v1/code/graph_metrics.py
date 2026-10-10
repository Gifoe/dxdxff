"""Exact repository graph definitions, deterministic solvers, explicit missingness.

Attribution: repository 2aacd6c11e_graph_channel.py. Local efficiency divides by
reachable ordered pairs (the repository convention), not all neighbor pairs.
No approximation or silent zero-valued solver failure is permitted.
"""
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata

BASE=['degree_norm','strength_norm','clustering_coeff','eigenvector_centrality','pagerank','kcore_norm','local_efficiency']
NAMES=[n+'__window_mean__seizure_mean' for n in BASE]+[n+'__window_std__seizure_mean' for n in BASE]+['temporal_strength_variation','cross_seizure_strength_rank_stability']

def node_features(a):
    a=np.asarray(a,dtype=np.float32)
    assert a.ndim==2 and a.shape[0]==a.shape[1] and np.isfinite(a).all()
    assert (a>=0).all() and np.array_equal(a,a.T) and not np.any(np.diag(a))
    n=len(a); degree=(a>0).sum(1); strength=a.sum(1)/max(n-1,1)
    clustering=np.zeros(n); eigen=np.zeros(n); core=np.zeros(n); local=np.zeros(n)
    pr=np.full(n,1/max(n,1),dtype=np.float64)
    if np.any(a):
        aa=a.astype(np.float64); root=np.cbrt(aa/aa.max()); denom=degree*(degree-1)
        clustering=np.divide(np.diag(root@root@root),denom,out=np.zeros(n),where=denom>0)
        values,vectors=eigh(aa,subset_by_index=[max(n-2,0),n-1],driver='evr',check_finite=False)
        if len(values)>1 and values[-1]-values[-2]<=1e-10*max(1.,abs(values[-1])):
            eigen[:]=np.nan # Dominant eigenspace not identifiable, not a zero fallback.
        else:
            eigen=vectors[:,-1]; eigen*=1 if eigen.sum()>=0 else -1
            assert np.min(eigen)>=-1e-8 and np.linalg.norm(aa@eigen-values[-1]*eigen)<=1e-8*max(1.,values[-1])
            eigen=np.maximum(eigen,0); eigen/=np.linalg.norm(eigen)
        rows=aa.sum(1); transition=np.divide(aa,rows[:,None],out=np.zeros_like(aa),where=rows[:,None]>0)
        for iteration in range(100):
            old=pr.copy(); pr=.85*(old@transition+old[rows==0].sum()/n)+.15/n
            if np.abs(pr-old).sum()<n*1e-6: break
        else: raise RuntimeError('PageRank solver did not converge; no silent fallback')
        graph=nx.from_numpy_array(aa); cd=nx.core_number(graph); maxcore=max(cd.values())
        core=np.array([cd[c]/max(maxcore,1) for c in range(n)])
        for c in range(n):
            neighbors=np.flatnonzero(a[c]>0)
            if len(neighbors)<2: continue
            sub=aa[np.ix_(neighbors,neighbors)]
            if not np.any(sub): continue
            distance=np.divide(1.,np.maximum(sub,1e-8),out=np.zeros_like(sub),where=sub>0)
            lengths=shortest_path(distance,directed=False,method='FW',overwrite=False)
            observed=np.isfinite(lengths)&(lengths>0)
            if observed.any(): local[c]=np.mean(1/lengths[observed])
    return np.stack([degree/max(n-1,1),strength,clustering,eigen,pr,core,local],axis=-1).astype(np.float32)

def seizure_summary(metrics,times):
    metrics=np.asarray(metrics,float); times=np.asarray(times,float)
    assert metrics.ndim==3 and metrics.shape[2]==7 and len(times)==len(metrics)
    count=np.isfinite(metrics).sum(0); mean=np.divide(np.nansum(metrics,axis=0),count,out=np.full(metrics.shape[1:],np.nan),where=count>0)
    variance=np.divide(np.nansum((metrics-mean)**2,axis=0),count,out=np.full_like(mean,np.nan),where=count>0)
    result=np.concatenate([mean,np.sqrt(variance),np.full((metrics.shape[1],1),np.nan)],axis=1)
    for c in range(metrics.shape[1]):
        ix=np.flatnonzero(np.isfinite(metrics[:,c,1])); ix=ix[np.argsort(times[ix],kind='stable')]
        if len(ix)>1: result[c,14]=np.mean(np.abs(np.diff(metrics[ix,c,1]))/np.maximum(np.diff(times[ix]),1.))
    rank=np.full(len(mean),np.nan); valid=np.isfinite(mean[:,1]); n=valid.sum()
    if n>1: rank[valid]=(rankdata(mean[valid,1],method='average')-1)/(n-1)
    elif n==1: rank[valid]=.5
    return result,rank

def patient_summary(summaries,ranks):
    s=np.asarray(summaries,float); rank=np.asarray(ranks,float); counts=np.isfinite(s).sum(0)
    out=np.divide(np.nansum(s,axis=0),counts,out=np.full(s.shape[1:],np.nan),where=counts>0)
    stability=np.full(s.shape[1],np.nan)
    for c in range(s.shape[1]):
        values=rank[:,c]; values=values[np.isfinite(values)]
        if len(values)>=2:
            i,j=np.triu_indices(len(values),1); stability[c]=np.mean(1-np.abs(values[i]-values[j]))
    return np.column_stack([out,stability])
