"""Fixed-label-free subsets; patient-level endpoints without classifiers."""
import hashlib
import math
import numpy as np
from scipy.special import ndtr

ARMS=['GLOBAL','SOURCE','CENTER','SCALE','FULL']
SOURCES=['hup','lzu','multicenter','pediatric']
COORDS=[f'{f}_{s}' for f in ['beta','low_gamma','partial_high_gamma','RMS','line_length','spectral_entropy'] for s in ['mean','SD']]
FRACTIONS=[.25,.5,.75,1.]

def crps(x,mu,sd):
    x,mu,sd=np.broadcast_arrays(np.asarray(x,float),np.asarray(mu,float),np.asarray(sd,float));assert np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>=0).all()
    a=np.abs(x-mu)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        u=np.divide(a,sd,out=np.zeros_like(a),where=sd>0)
        result=a*(2*ndtr(u)-1)+sd*(np.sqrt(2/np.pi)*np.exp(-u*u/2)-1/np.sqrt(np.pi))
    result=np.where(sd==0,a,result);assert np.isfinite(result).all();return result

def moments(x,axis=0):
    mu=x.mean(axis);sd=x.std(axis,ddof=0);sd=np.where(np.ptp(x,axis=axis)==0,0.,sd)
    return np.stack([mu,sd])

def rankings(patient,channels):
    return np.array([sorted(range(len(channels)),key=lambda i:(hashlib.sha256(f'42|PRREFV1|{patient}|{r}|{channels[i]}'.encode()).digest(),channels[i])) for r in range(100)],dtype=np.int32)

def nested_indices(order):
    n=len(order);a=np.broadcast_to(order,(n,n));others=a[a!=np.arange(n)[:,None]].reshape(n,n-1)
    return [others[:,:math.ceil(f*(n-1))].copy() for f in FRACTIONS]

def freeze_patient(p):
    x=p['x'];order=rankings(p['patient'],p['channels']);params=[]
    full=np.array([np.flatnonzero(np.arange(len(x))!=q) for q in range(len(x))]);full_parameters=moments(x[full],axis=1)
    for indices in order:
        refs=nested_indices(indices);params.append([moments(x[q],axis=1) for q in refs[:3]]+[full_parameters])
    return dict(order=order,parameters=np.array(params),reference_counts=np.array([math.ceil(f*(len(x)-1)) for f in FRACTIONS]))

def class_mean(x,y,axis=0):
    assert set(y)=={0,1};return np.stack([np.take(x,np.flatnonzero(y==v),axis=axis).mean(axis=axis) for v in [1,0]])

def e1_scores(p):
    x=p['x'];out=np.zeros((len(x),5,12));oldabs=np.zeros_like(x)
    for r,mask in enumerate(p['masks']):
        for j,ref in enumerate([mask,~mask]):
            q=~ref;mu,sd=p['references'][r,j]
            for a,(loc,s) in enumerate([p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],[mu,sd]]):out[q,a]+=crps(x[q],loc,s)/100
            oldabs[q]+=np.abs(np.divide(x[q]-mu,sd,out=np.zeros_like(x[q]),where=sd>1e-8))/100
    return out,oldabs

def h_from_class(loss):
    gain=loss[...,0:1,:]-loss
    return gain[1]-gain[0]

def label_null(score,y,patient):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'42|PRREFLABEL|{patient}'.encode()).digest()[:16],'little'))
    gains=score[:,0:1,:].mean(-1)-score.mean(-1);out=[]
    for _ in range(1000):
        z=rng.permutation(y);out.append(gains[z==0].mean(0)-gains[z==1].mean(0))
    return np.array(out)

def composition_slope(u,v):
    u=u-u.mean(axis=0,keepdims=True);v=v-v.mean(axis=0,keepdims=True)
    denom=float(np.square(u).sum())
    return (.1*float((u*v).sum())/denom if denom>1e-20 else np.nan),denom

def evaluate_patient(p,frozen,y):
    x=p['x'];n=len(x);e1,legacy=e1_scores(p)
    losses=np.zeros((4,n,5,12));absolute=np.zeros((4,n,12));support=np.zeros((4,n,12));unsupported_nonconstant=np.zeros_like(support)
    composition=np.zeros((100,4,n));full_loss=np.zeros((100,4,n))
    for r,order in enumerate(frozen['order']):
        refs=nested_indices(order)
        for f,ix in enumerate(refs):
            mu,sd=frozen['parameters'][r,f]
            for a,(loc,s) in enumerate([p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],[mu,sd]]):
                current=crps(x,loc,s);losses[f,:,a]+=current/100
                if a==4:full_loss[r,f]=current.mean(-1)
            absolute[f]+=np.abs(np.divide(x-mu,sd,out=np.zeros_like(x),where=sd>1e-8))/100
            support[f]+=(sd<=1e-8)/100
            unsupported_nonconstant[f]+=((sd<=1e-8)&(np.abs(x-mu)>1e-8))/100
            composition[r,f]=y[ix].mean(1)
    slopes=[];denoms=[]
    for f in range(3):
        b,d=composition_slope(composition[:,f],full_loss[:,f]);slopes.append(b);denoms.append(d)
    e1class=class_mean(e1,y)
    e2class=np.array([class_mean(q,y) for q in losses]);absclass=np.array([class_mean(q,y) for q in absolute])
    return dict(e1_class=e1class,e1_all=e1.mean(0),e1_h=h_from_class(e1class),legacy_abs_class=class_mean(legacy,y),
        e1_null=label_null(e1,y,p['patient']),e2_all=losses.mean(1),e2_class=e2class,e2_h=np.array([h_from_class(q) for q in e2class]),e2_abs_class=absclass,e2_abs_all=absolute.mean(1),
        support=support.mean(1),unsupported_nonconstant=unsupported_nonconstant.mean(1),composition_slope=np.array(slopes),composition_denominator=np.array(denoms),
        composition_mean=composition.mean((0,2)),composition_min=composition.min((0,2)),composition_max=composition.max((0,2)),reference_counts=frozen['reference_counts'])

def bootstrap_indices(src):
    src=np.asarray(src);rng=np.random.default_rng(42)
    return np.concatenate([rng.choice(np.flatnonzero(src==s),size=(10000,int((src==s).sum())),replace=True) for s in SOURCES],axis=1)
