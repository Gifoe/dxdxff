"""Label-free transformations: patient moments, FIT global moments, derangement."""
from source import *
def patient_moments(h):
    h=np.asarray(h,np.float32);return h.mean(0),np.sqrt(h.var(0)+np.float32(1e-5))
def global_moments(data,patients):
    assert patients and len(set(patients))==len(patients)
    mean=np.mean([data[p]['h'].astype(np.float64).mean(0) for p in patients],axis=0)
    variance=np.mean([((data[p]['h'].astype(np.float64)-mean)**2).mean(0) for p in patients],axis=0)
    assert np.isfinite(mean).all() and np.isfinite(variance).all() and (variance>=0).all()
    return {'mean':mean,'variance':variance,'scale':np.sqrt(variance+1e-5),'fit_patient_digest':digest(sorted(patients)),'fit_patients':len(patients)}
def permutation(fold,role,pseudonym,n):
    assert role in ['FIT','HELD'] and n>=2
    rng=np.random.default_rng(seeded(fold,role,pseudonym,'embedding-derangement'))
    for attempt in range(1024):
        ix=rng.permutation(n)
        if not np.any(ix==np.arange(n)):return ix
    raise RuntimeError('Labelblind derangement failed; no outcome-based retry')
def transform(data,patients,arm,fold,role,scaler=None):
    assert arm in ['C1-PR','C1-ABS','C1-SHUFFLE'];result={}
    for p in patients:
        d=data[p];x=dict(d)
        if arm=='C1-ABS':
            assert scaler is not None;x['z']=((d['h'].astype(np.float64)-scaler['mean'])/scaler['scale']).astype(np.float32)
        else:
            x['z']=normalized(d['h'])
            if arm=='C1-SHUFFLE':
                ix=permutation(fold,role,key(p),len(x['z']));old=x['z'];x['z']=old[ix]
                assert np.array_equal(x['z'][np.argsort(ix)],old)
                x['permutation']=ix
        assert np.isfinite(x['z']).all();result[p]=x
    return result
def scale_diagnostic(h,base,head,source):
    """Direct float64 formula; historical float32 moments remain fixed."""
    mean,scale=patient_moments(h);beta=head.beta.detach().numpy().astype(np.float64);bias=float(head.b[source].detach())
    absolute=h.astype(np.float64)/scale.astype(np.float64)
    centered=(h.astype(np.float64)-mean.astype(np.float64))/scale.astype(np.float64)
    reference=base+centered@beta+bias;unscaled=base+absolute@beta+bias
    offset=float((mean.astype(np.float64)/scale.astype(np.float64))@beta)
    error=float(np.max(abs((unscaled-reference)-offset)))
    i,j=np.triu_indices(len(base),1)
    relation=lambda v:np.sign(v[i]-v[j])
    assert error<1e-10 and np.array_equal(relation(reference),relation(unscaled)), 'Scale-only mathematical ordering failure'
    return reference,unscaled,offset,error
