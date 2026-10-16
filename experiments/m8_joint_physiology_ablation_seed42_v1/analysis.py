"""Frozen contrast algebra and validation-only candidate rule; no torch/data loaders."""
import itertools
import numpy as np

CONFIGS = [''.join(map(str, x)) for x in itertools.product([0, 1], repeat=3)]
FACTORS = {'V': 0, 'H': 1, 'E': 2}
GROUPS = [4, 2, 6]
METRICS = ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']

def keep(config):
    assert config in CONFIGS
    out = np.ones(7, dtype=np.float32)
    for bit, group in zip(config, GROUPS):
        if bit == '1': out[group] = 0
    return out.tolist()

def factorial_weights(factors, condition=None):
    axes = [FACTORS[x] for x in factors]
    fixed = {} if condition is None else {FACTORS[k]:v for k,v in condition.items()}
    assert not set(axes) & set(fixed)
    free = 3-len(axes)-len(fixed)
    weights = {}
    for c in CONFIGS:
        if any(int(c[k]) != v for k,v in fixed.items()): continue
        weights[c] = np.prod([1 if c[k]=='1' else -1 for k in axes]) / 2**free
    return weights

def bootstrap_counts(ids, draws=10000):
    unique = sorted(set(ids)); draw=np.random.default_rng(42).integers(0,len(unique),size=(draws,len(unique)))
    counts=np.stack([(draw==i).sum(1) for i in range(len(unique))],1)
    return unique,counts,draw

def contrast(matrix, weights, ids, counts, unique, alpha=.05):
    """matrix keys -> [patient-appearance,metric]; complete-case metric contrasts."""
    keys=list(weights); values=np.stack([matrix[k] for k in keys])
    valid=np.isfinite(values).all(0)
    delta=sum(float(weights[k])*matrix[k] for k in keys)
    delta=np.where(valid,delta,np.nan)
    byid=np.stack([np.nansum(delta[np.asarray(ids)==p],axis=0) for p in unique])
    denid=np.stack([np.isfinite(delta[np.asarray(ids)==p]).sum(0) for p in unique])
    den=counts@denid
    distribution=np.divide(counts@byid,den,out=np.full(den.shape,np.nan),where=den>0)
    mean=np.divide(np.nansum(delta,axis=0),valid.sum(0),out=np.full(delta.shape[1],np.nan),where=valid.sum(0)>0)
    return mean,np.nanpercentile(distribution,[100*alpha/2,100*(1-alpha/2)],axis=0),valid.sum(0),distribution

def candidate_decisions(patients, protocol):
    """No Outer input accepted: requires explicit development scope."""
    assert set(patients['scope']) == {'development'}
    rules=protocol['selection']; rows=[]; choices={}
    for head in ['D1','M8']:
        ref=patients[(patients['head']==head)&(patients['config']=='000')].sort_values(['fold','patient'])
        unique,counts,_=bootstrap_counts(ref.patient.to_numpy())
        order=list(zip(ref.fold,ref.patient)); matrices={}
        for config in CONFIGS:
            frame=patients[(patients['head']==head)&(patients['config']==config)].sort_values(['fold','patient'])
            assert list(zip(frame.fold,frame.patient))==order
            matrices[config]=frame[METRICS].to_numpy()
        eligible=[]
        for config in rules['eligible_candidates']:
            g=patients[(patients['head']==head)&(patients['config']==config)].sort_values(['fold','patient'])
            diff=matrices[config]-matrices['000']; fold_d=[]; fold_ok=[]
            for f in range(1,6):
                d=np.nanmean(diff[g.fold.to_numpy()==f],axis=0); fold_d.append(d)
                fold_ok.append(all(d[METRICS.index(k)]>=v for k,v in rules['per_fold_tolerances'].items()))
            sources=[]
            for source in sorted(set(ref.center)):
                ix=ref.center.to_numpy()==source
                if ix.sum()>=rules['source_min_appearances']:
                    sources.append(float(np.nanmean(diff[ix,METRICS.index('macro_f1')])))
            mean,ci,n,_=contrast(matrices,{config:1,'000':-1},ref.patient.to_numpy(),counts,unique)
            fd=np.asarray(fold_d)[:,METRICS.index('macro_f1')]
            ok=(sum(fold_ok)>=rules['minimum_eligible_folds'] and np.median(fd)>=rules['median_fold_macro_delta_min']
                and fd.min()>=rules['worst_fold_macro_delta_min'] and all(x>=rules['source_macro_delta_min'] for x in sources)
                and all(ci[0,METRICS.index(k)]>v for k,v in rules['cluster95_lower_bounds_strictly_greater'].items()))
            row=dict(head=head,config=config,retained_groups=int(sum(keep(config))),eligible=bool(ok),eligible_folds=int(sum(fold_ok)),
                median_fold_macro_delta=float(np.median(fd)),worst_fold_macro_delta=float(fd.min()),
                worst_source_macro_delta=min(sources) if sources else None,
                **{'delta_'+k:float(mean[i]) for i,k in enumerate(METRICS)},
                **{'lower95_'+k:float(ci[0,i]) for i,k in enumerate(METRICS)})
            rows.append(row)
            if ok: eligible.append((row,g))
        eligible.sort(key=lambda x:(x[0]['retained_groups'],-float(x[1].macro_f1.mean()),-float(x[1].auprc.mean()),-float(x[1].auroc.mean()),x[0]['config']))
        choices[head]=eligible[0][0]['config'] if eligible else '000'
    return rows,choices
