"""Changed critical path only: decisions/calibration, not a fitted outcome."""
import inspect
import numpy as np
from sklearn.metrics import f1_score
from common import probability
from decoders import predict,fit,candidates,utility,score_z,oracle_macro


def test_binary_ties_and_permutations():
    x=np.array([-3.,-3.,-3.,1.,1.,4.]);p=np.repeat('p',6);order=np.array([4,0,5,3,1,2])
    for family in ['MARGIN_GRID','PATIENT_Z','TOP_FRACTION']:
        for c in candidates(family,x):
            a=predict(c,x,p);b=predict(c,x[order],p[order])
            assert np.array_equal(a[order],b)
            assert len(set(a[:3]))==len(set(a[3:5]))==1
    assert 'y' not in inspect.signature(predict).parameters


def test_constant_population_z_and_original_precision():
    x=np.r_[np.repeat(20.,4),-2.];p=np.array(['p']*4+['q'])
    assert np.array_equal(score_z(x,p),np.zeros(5))
    c=dict(family='ORIGINAL',value=.5)
    assert np.array_equal(predict(c,x,p),probability(x)<.5)
    assert probability(np.array([20.,30.]))[0]==probability(np.array([20.,30.]))[1]
    assert 20.<30.  # Raw margins retain order even when FP32 probability saturates.
    rng=np.random.default_rng(42);x=rng.normal(size=73);p=np.repeat('p',73)
    z=score_z(x,p);assert abs(z.mean())<1e-12 and abs(z.std()-1)<1e-12
    assert np.max(np.abs(z))<=np.sqrt(len(z)-1)+1e-10


def test_vectorized_utility_and_independent_calibration():
    x=np.array([-4.,-4.,-.2,.5,2.,3.,-3.,-1.,0.,1.,2.,5.]);p=np.repeat(['p','q'],6)
    y=np.array([1,0,1,0,0,0,1,1,0,0,0,1]);original=dict(family='ORIGINAL',value=.5)
    for family in ['MARGIN_GRID','PATIENT_Z','TOP_FRACTION']:
        configs=candidates(family,x);keys=[]
        for i,c in enumerate(configs):
            pr=predict(c,x,p);m=[];e=[];change=[]
            for pid in ['p','q']:
                j=p==pid;m.append(f1_score(y[j],pr[j],average='macro',zero_division=0))
                e.append(f1_score(y[j],pr[j],zero_division=0));change.append(np.mean(pr[j]!=predict(original,x,p)[j]))
                a,b=utility(y[j],pr[j]);assert abs(a[0]-m[-1])<1e-12 and abs(b[0]-e[-1])<1e-12
            v=c['value'];distance=abs(v-(.5 if family=='TOP_FRACTION' else 0)) if v is not None else float('inf')
            keys.append((np.mean(m),np.mean(e),-np.mean(change),-distance,-i))
        expected=configs[max(range(len(keys)),key=keys.__getitem__)]
        selected,_=fit(family,x,1-y,p,original);assert selected==expected


def test_oracle_is_diagnostic_and_bound_on_thresholds():
    x=np.array([-2.,-1.,-1.,2.,5.]);y=np.array([1,0,1,0,1]);p=np.repeat('p',5)
    bound=oracle_macro(y,x)
    for c in candidates('MARGIN_GRID',x):
        assert f1_score(y,predict(c,x,p),average='macro',zero_division=0)<=bound+1e-12
    assert 'oracle' not in inspect.getsource(fit)

