"""Independent reconstruction: alternate CRPS,reference arithmetic,bootstrap counts."""
import hashlib
import math
import numpy as np
from scipy.special import erfc
import contract as io

def crps_other(x,mu,sigma):
    diff=np.abs(np.asarray(x,dtype=float)-mu);s=np.broadcast_to(np.asarray(sigma,float),diff.shape)
    pos=s>0;out=diff.copy()
    with np.errstate(over='ignore',under='ignore'):
        u=diff[pos]/s[pos]
        out[pos]=diff[pos]*(1-erfc(u/np.sqrt(2)))+s[pos]/np.sqrt(np.pi)*(np.sqrt(2)*np.exp(-u*u/2)-1)
    return out

def mean_sd(x):
    mu=np.sum(x,axis=0)/x.shape[0];v=np.sum((x-mu)**2,axis=0)/x.shape[0]
    sd=np.sqrt(v);sd[np.max(x,axis=0)==np.min(x,axis=0)]=0
    return np.stack([mu,sd])

def replay(p,y):
    x=p['x'];n=len(x);forecasts=np.zeros((5,12));deviation=np.zeros_like(x);support=np.zeros(12);cross=np.zeros(12)
    for r in range(100):
        keys=[hashlib.sha256(f'42|PRREFV1|{p["patient"]}|{r}|{c}'.encode()).hexdigest() for c in p['channels']]
        order=sorted(range(n),key=lambda i:(keys[i],p['channels'][i]));a=np.array(order[:n//2]);b=np.array(order[n//2:])
        assert set(a).isdisjoint(b) and set(a)|set(b)==set(range(n))
        assert np.array_equal(np.sort(a),np.flatnonzero(p['masks'][r]))
        for side,(ref,q) in enumerate([(a,b),(b,a)]):
            rs=mean_sd(x[ref]);np.testing.assert_allclose(rs,p['references'][r,side],rtol=1e-12,atol=1e-12)
            mu,sd=rs
            models=[p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],rs]
            for j,(location,scale) in enumerate(models):forecasts[j]+=crps_other(x[q],location,scale).sum(0)/(100*n)
            valid=sd>1e-8;z=np.zeros_like(x[q]);z[:,valid]=np.abs((x[q][:,valid]-mu[valid])/sd[valid]);deviation[q]+=z/100
            support+=(~valid)*len(q)/(100*n)
            cross+=((~valid)&(np.abs(x[q]-mu)>1e-8)).sum(0)/(100*n)
    ez=deviation[y==1].sum(0)/int(y.sum());nez=deviation[y==0].sum(0)/int((y==0).sum())
    seed=int.from_bytes(hashlib.sha256(('42|PRREFLABEL|'+p['patient']).encode()).digest()[:16],byteorder='little')
    rng=np.random.default_rng(seed);score=deviation.sum(1)/12;null=[]
    for _ in range(1000):
        labels=y[rng.permutation(n)];null.append(score[labels==1].sum()/y.sum()-score[labels==0].sum()/(n-y.sum()))
    return dict(crps=forecasts,deviation=deviation,EZ=ez,NEZ=nez,unsupported=support,unsupported_query_nonconstant=cross,permutation=np.array(null))

def verify(root):
    data,binding=io.checked(root);freeze=io.read(root/'public/FREEZE.json');seal=io.read(root/'public/SEAL.json')
    assert io.sha(root/'private/REFERENCES.pkl')==freeze['references_sha256']
    assert io.sha(root/'private/ENDPOINTS.pkl')==seal['sha256']
    plans=io.load(root/'private/REFERENCES.pkl');rows=io.load(root/'private/ENDPOINTS.pkl');assert len(rows)==len(plans)==80
    labels={};verified=[];maxerr=0.;values=0
    # Refit legal global/source transforms independently of frozen parameters.
    for d in data:
        tr=d['raw'][d['train']];mu,sd=mean_sd(tr);unit=np.where(sd>0,sd,1);x=(d['raw']-mu)/unit
        for p in plans:
            if p['fold']!=d['fold']:continue
            ix=np.flatnonzero(d['patient']==p['patient']);labels[p['patient']]=d['y'][ix]
            np.testing.assert_allclose(x[ix],p['x'],rtol=1e-10,atol=1e-11)
            np.testing.assert_allclose(mean_sd(x[d['train']]),p['global'],rtol=1e-10,atol=1e-11)
            np.testing.assert_allclose(mean_sd(x[d['train']][d['source'][d['train']]==p['source']]),p['source_stats'],rtol=1e-10,atol=1e-11)
    for i,(p,r) in enumerate(zip(plans,rows)):
        assert p['patient']==r['patient'];q=replay(p,labels[p['patient']])
        for k,v in q.items():
            np.testing.assert_allclose(v,r[k],rtol=1e-9,atol=1e-10,err_msg=k)
            maxerr=max(maxerr,float(np.max(np.abs(v-r[k]))));values+=np.size(v)
        verified.append(dict(q,source=p['source'],patient=p['patient']))
        if (i+1)%10==0:print('INDEPENDENT_PATIENTS',i+1,flush=True)
    # Table implementation checked against independently weighted bootstrap counts.
    import stats_report as report
    tables=report.summarize(rows)
    src=np.array([p['source'] for p in plans]);rng=np.random.default_rng(42);groups=['hup','lzu','multicenter','pediatric']
    sampled=np.concatenate([np.flatnonzero(src==s)[rng.integers(0,int((src==s).sum()),size=(10000,int((src==s).sum())))] for s in groups],axis=1)
    counts=np.array([np.bincount(q,minlength=80) for q in sampled]);assert (counts.sum(1)==80).all()
    arrays={k:np.array([r[k] for r in verified]) for k in ['crps','EZ','NEZ','unsupported','unsupported_query_nonconstant','permutation']};arrays['gap']=arrays['EZ']-arrays['NEZ']
    arms=['GLOBAL','SOURCE','CENTER','SCALE','FULL'];names=[f'{f}_{s}' for f in ['beta','low_gamma','partial_high_gamma','RMS','line_length','spectral_entropy'] for s in ['mean','SD']]
    tested=family=0
    for table in ['BACKGROUND','CONTRASTS','ANOMALY','SUPPORT']:
        for row in tables[table]:
            idx=np.arange(80) if row['scope']=='ALL' else np.flatnonzero(src==row['scope'])
            j=None if row['coordinate']=='ALL12' else names.index(row['coordinate'])
            if table in ['BACKGROUND','CONTRASTS']:
                a=arrays['crps'][idx];a=a.mean(2) if j is None else a[:,:,j]
                if table=='BACKGROUND':v=a[:,arms.index(row['arm'])]
                else:
                    left,right=row['contrast'].split('-');v=a[:,arms.index(left)]-a[:,arms.index(right)]
            else:
                a=arrays[row['endpoint']][idx];v=a.mean(1) if j is None else a[:,j]
            boot=(counts[:,idx]@v)/len(idx);expected=dict(mean=float(np.mean(v)),median=float(np.median(v)),positive_fraction=float(np.mean(v>0)))
            expected.update(zip(['ci_low','ci_high'],np.percentile(boot,[2.5,97.5])))
            if 'family_low' in row:
                family+=1;expected.update(zip(['family_low','family_high'],np.percentile(boot,[100*.05/12,100*(1-.05/12)])))
            for key,val in expected.items():np.testing.assert_allclose(val,row[key],rtol=1e-9,atol=1e-10)
            tested+=1
    for row in tables['PERMUTATION']:
        idx=np.arange(80) if row['scope']=='ALL' else np.flatnonzero(src==row['scope'])
        np.testing.assert_allclose(arrays['permutation'][idx,row['permutation']].mean(),row['gap'],rtol=1e-9,atol=1e-10)
    assert family==6 and sum(len(p['x']) for p in plans)==7635
    io.write(root/'public/INDEPENDENT_VERIFICATION.json',dict(status='PASS',patients=80,channels=7635,partitions=8000,crps_patient_arm_coordinate_values=4800,endpoint_values_checked=values,aggregate_interval_rows_checked=tested,family_intervals_checked=family,permutation_aggregate_values_checked=5000,max_endpoint_absolute_error=maxerr,absolute_tolerance=1e-10,relative_tolerance=1e-9,frozen_sources_unchanged=True,source_commit=io.commit()))
    print('INDEPENDENT_VERIFICATION_PASS',flush=True)
