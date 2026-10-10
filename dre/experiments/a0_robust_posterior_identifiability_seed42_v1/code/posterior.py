"""Label-free patient inference; labels are accepted only by FIT density fitting."""
import numpy as np
from scipy.special import expit

class DensityInvalid(ValueError):
    def __init__(self, audit):
        self.audit=audit
        super().__init__('POSTERIOR_DENSITY_INVALID')

def iqr(logits):
    logits=np.asarray(logits,dtype=np.float64)
    return float(np.quantile(logits,.75)-np.quantile(logits,.25)) if len(logits) else 0.

def fit_floor(logits,patient,allowed_fit):
    assert set(patient)==set(allowed_fit)
    values=[iqr(logits[patient==p]) for p in sorted(set(patient))]
    positive=[v for v in values if np.isfinite(v) and v>0]
    if not positive:raise ValueError('R1_NO_POSITIVE_FIT_IQR')
    return max(1e-3,float(np.quantile(positive,.10)))

def normalize(logits,representation,floor=None):
    logits=np.asarray(logits,dtype=np.float64)
    assert representation in ['R0','R1'] and np.isfinite(logits).all()
    r=iqr(logits)
    if representation=='R0':return logits.copy(),r
    assert floor is not None and floor>=1e-3 and np.isfinite(floor)
    if not len(logits):return logits.copy(),r
    return np.clip((logits-np.median(logits))/max(r,floor),-8.,8.),r

def fit_density(logits,y,patient,g,allowed_fit,representation,floor=None):
    """Patient-equal class moments with unique-patient hierarchical shrinkage."""
    assert set(patient)==set(allowed_fit), 'Density fitting must cover exactly outer FIT'
    episodes=[]
    for p in sorted(set(patient)):
        ix=np.flatnonzero(patient==p); z,r=normalize(logits[ix],representation,floor); gg=np.unique(g[ix])
        assert len(gg)==1 and 0<=gg[0]<4
        row={'g':int(gg[0]),'prior':float(np.mean(y[ix]==0)), 'low_iqr':r<=1e-5}
        for cls in [0,1]:
            v=z[y[ix]==cls]
            row[str(cls)]=None if not len(v) else (float(v.mean()),float(np.mean(v*v)))
        episodes.append(row)
    def moments(rows):
        out={}
        for c in ['0','1']:
            values=[r[c] for r in rows if r[c] is not None]
            out[c]={'n':len(values),'mean':float(np.mean([v[0] for v in values])) if values else np.nan,
                    'second':float(np.mean([v[1] for v in values])) if values else np.nan}
        return out
    global_m=moments(episodes)
    def density(m):
        return {'mu_ez':m['0']['mean'],'mu_nez':m['1']['mean'],
            'variance':max(1e-6,float(np.mean([m[c]['second']-m[c]['mean']**2 for c in ['0','1']])))}
    glob=density(global_m)
    if min(global_m[c]['n'] for c in ['0','1'])<3 or not glob['mu_ez']<glob['mu_nez']:
        raise DensityInvalid({'global':glob,'global_moments':global_m,'fit_patients':len(episodes)})
    pi=float(np.mean([r['prior'] for r in episodes])); glob.update(pi=pi,fallback=False)
    sources=[]
    for group in range(4):
        rows=[r for r in episodes if r['g']==group]; raw=moments(rows); shrunk={}
        for c in ['0','1']:
            n=raw[c]['n']; w=n/(n+10)
            shrunk[c]={'mean':w*(raw[c]['mean'] if n else 0)+(1-w)*global_m[c]['mean'],
                       'second':w*(raw[c]['second'] if n else 0)+(1-w)*global_m[c]['second']}
        d=density(shrunk); n=len(rows)
        sp=(n*np.mean([r['prior'] for r in rows])+10*pi)/(n+10) if n else pi
        reason='none'
        if min(raw[c]['n'] for c in ['0','1'])<3:reason='insufficient_class_bearing_patients'
        elif not raw['0']['mean']<raw['1']['mean']:reason='raw_reversed_order'
        elif not d['mu_ez']<d['mu_nez']:reason='shrunk_reversed_order'
        if reason!='none':d={k:glob[k] for k in ['mu_ez','mu_nez','variance']}
        d.update(pi=float(sp),fallback=reason!='none',fallback_reason=reason,patients=n,
                 class_patients=[raw[c]['n'] for c in ['0','1']],raw_moments=raw,shrunk_moments=shrunk)
        sources.append(d)
    return {'global':glob,'sources':sources,'global_moments':global_m,
            'fit_patients':len(episodes),'low_iqr_fraction':float(np.mean([r['low_iqr'] for r in episodes]))}

def log_densities(z,d):
    var=d['variance']; assert var>0 and d['mu_ez']<d['mu_nez']
    constant=-.5*np.log(2*np.pi*var)
    return constant-.5*(z-d['mu_ez'])**2/var,constant-.5*(z-d['mu_nez'])**2/var

def infer(logits,source,density,representation,floor=None,fixed=False):
    """No labels, counts, patient IDs, outcome metadata or VAL fitting argument."""
    z,r=normalize(logits,representation,floor)
    d=density['sources'][source] if 0<=source<4 else density['global']
    le,ln=log_densities(z,d); prior=d['pi']; pi=prior; iterations=0; converged=True
    def derivative(p):
        q=expit(np.log(p)-np.log1p(-p)+le-ln)
        return float(np.sum(q/p-(1-q)/(1-p))+20*prior/p-20*(1-prior)/(1-p))
    if not fixed and len(z):
        lo,hi=1e-6,1-1e-6
        if derivative(lo)<=0:pi=lo
        elif derivative(hi)>=0:pi=hi
        else:
            for iterations in range(1,101):
                mid=(lo+hi)/2
                if derivative(mid)>0:lo=mid
                else:hi=mid
                if hi-lo<=1e-12:break
            pi=(lo+hi)/2; converged=hi-lo<=1e-12
        if not converged:pi=prior
    logodds=np.log(pi)-np.log1p(-pi)+le-ln
    q=expit(logodds)
    def ll(p):return float(np.sum(np.logaddexp(np.log(p)+le,np.log1p(-p)+ln)))
    assert np.isfinite(le).all() and np.isfinite(ln).all() and np.isfinite(pi) and np.isfinite(ll(pi)) and np.isfinite(ll(prior))
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    return q,{'pi':float(pi),'prior':float(prior),'ll_gain':ll(pi)-ll(prior),
        'iterations':iterations,'converged':converged,'near_boundary':pi<=1e-4 or pi>=1-1e-4,
        'low_iqr':r<=1e-5,'fallback':bool(d['fallback']),'logodds':logodds,
        'separation':(d['mu_nez']-d['mu_ez'])/np.sqrt(d['variance'])}

def decode(q,channels,logits=None):
    q=np.asarray(q,dtype=np.float64); C=len(q)
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    if C==0:return np.zeros(0,bool),{'k':0,'objective':0.,'expected_count':0.,'all_k':np.array([0.])}
    # q saturation must not replace the mathematically monotonic ordering.
    order=np.lexsort((np.asarray(channels).astype(str),-q)) if logits is None else np.lexsort((np.asarray(channels).astype(str),np.asarray(logits)))
    if logits is not None:assert np.all(np.diff(q[order])<=1e-13)
    tp=np.r_[0.,np.cumsum(q[order])]; k=np.arange(C+1,dtype=float)
    fp=k-tp; fn=tp[-1]-tp; tn=C-k-fn
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a),where=b>0)
    values=.5*(divide(2*tp,2*tp+fp+fn)+divide(2*tn,2*tn+fp+fn))
    selected=int(np.argmax(values)); pred_ez=np.zeros(C,bool); pred_ez[order[:selected]]=True
    return pred_ez,{'k':selected,'objective':float(values[selected]),'expected_count':float(tp[-1]),'all_k':values}

def calibration(q,y):
    target=np.asarray(y)==0; q=np.asarray(q); brier=float(np.mean((q-target)**2))
    bins=np.minimum((q*10).astype(int),9); ece=0.
    for b in range(10):
        take=bins==b
        if take.any():ece+=take.mean()*abs(q[take].mean()-target[take].mean())
    return brier,float(ece)
