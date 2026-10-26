"""Direct reference replay,alternative CRPS andindependent bootstrap arithmetic."""
import hashlib
import math
import numpy as np
from scipy.special import erfc
import io_contract as io

def crps_other(x,mu,sigma):
    a=np.abs(np.asarray(x,float)-mu);s=np.broadcast_to(np.asarray(sigma,float),a.shape);out=a.copy();good=s>0
    with np.errstate(over='ignore',under='ignore'):
        z=a[good]/s[good];out[good]=a[good]*(1-erfc(z/np.sqrt(2)))+s[good]*(np.sqrt(2/np.pi)*np.exp(-z*z/2)-1/np.sqrt(np.pi))
    return out

def moment_other(x):
    mean=np.sum(x,axis=0)/len(x);sd=np.sqrt(np.sum((x-mean)**2,axis=0)/len(x));sd[np.max(x,axis=0)==np.min(x,axis=0)]=0
    return mean,sd

def replay(p,y,frozen):
    x=p['x'];n=len(x);s1=np.zeros((n,5,12));s2=np.zeros((4,n,5,12));ab0=np.zeros_like(x);ab2=np.zeros((4,n,12));support=np.zeros((4,12));cross=np.zeros((4,12));u=np.zeros((100,4,n));v=np.zeros_like(u)
    for r in range(100):
        keys=[hashlib.sha256(f'42|PRREFV1|{p["patient"]}|{r}|{c}'.encode()).hexdigest() for c in p['channels']]
        order=sorted(range(n),key=lambda q:(keys[q],p['channels'][q]));assert np.array_equal(order,frozen['order'][r])
        for side,ref in enumerate([order[:n//2],order[n//2:]]):
            mu,sd=moment_other(x[ref]);np.testing.assert_allclose([mu,sd],p['references'][r,side],rtol=1e-9,atol=1e-10)
            for q in sorted(set(range(n))-set(ref)):
                for arm,(loc,scale) in enumerate([p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],[mu,sd]]):s1[q,arm]+=crps_other(x[q],loc,scale)/100
                z=np.zeros(12);ok=sd>1e-8;z[ok]=np.abs((x[q,ok]-mu[ok])/sd[ok]);ab0[q]+=z/100
        for q in range(n):
            available=[j for j in order if j!=q]
            for f,rho in enumerate([.25,.5,.75,1.]):
                ref=available[:math.ceil(rho*(n-1))]
                if f==3:ref=sorted(ref)
                assert q not in ref and len(ref)==frozen['reference_counts'][f]
                mu,sd=moment_other(x[ref]);np.testing.assert_allclose([mu,sd],frozen['parameters'][r,f,:,q],rtol=1e-9,atol=1e-10)
                for arm,(loc,scale) in enumerate([p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],[mu,sd]]):
                    score=crps_other(x[q],loc,scale);s2[f,q,arm]+=score/100
                    if arm==4:v[r,f,q]=sum(score)/12
                ok=sd>1e-8;z=np.zeros(12);z[ok]=np.abs((x[q,ok]-mu[ok])/sd[ok]);ab2[f,q]+=z/100
                support[f]+=(~ok)/(100*n);cross[f]+=((~ok)&(np.abs(x[q]-mu)>1e-8))/(100*n);u[r,f,q]=sum(y[ref])/len(ref)
    c1=np.array([s1[y==g].sum(0)/int((y==g).sum()) for g in [1,0]])
    c2=np.array([[s2[f,y==g].sum(0)/int((y==g).sum()) for g in [1,0]] for f in range(4)])
    def h(loss):return (loss[1,0]-loss[1])-(loss[0,0]-loss[0])
    slopes=[];denoms=[]
    for f in range(3):
        num=den=0.
        for q in range(n):
            xx=u[:,f,q]-sum(u[:,f,q])/100;yy=v[:,f,q]-sum(v[:,f,q])/100;num+=float(xx@yy);den+=float(xx@xx)
        slopes.append(.1*num/den if den>1e-20 else np.nan);denoms.append(den)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('42|PRREFLABEL|'+p['patient']).encode()).digest()[:16],'little'))
    null=[];gains=(s1[:,0,None]-s1).sum(2)/12
    for _ in range(1000):
        z=y[rng.permutation(n)];null.append(gains[z==0].sum(0)/(n-y.sum())-gains[z==1].sum(0)/y.sum())
    return dict(e1_class=c1,e1_all=s1.sum(0)/n,e1_h=h(c1),legacy_abs_class=np.array([ab0[y==g].sum(0)/int((y==g).sum()) for g in [1,0]]),e1_null=np.array(null),e2_all=s2.sum(1)/n,e2_class=c2,e2_h=np.array([h(z) for z in c2]),e2_abs_class=np.array([[ab2[f,y==g].sum(0)/int((y==g).sum()) for g in [1,0]] for f in range(4)]),e2_abs_all=ab2.sum(1)/n,support=support,unsupported_nonconstant=cross,composition_slope=np.array(slopes),composition_denominator=np.array(denoms),composition_mean=u.mean((0,2)),composition_min=u.min((0,2)),composition_max=u.max((0,2)),reference_counts=np.array([math.ceil(t*(n-1)) for t in [.25,.5,.75,1.]]))

def independent_table_check(rows,tables):
    from core import SOURCES,COORDS,ARMS
    src=np.array([p['source'] for p in rows]);rng=np.random.default_rng(42)
    draw=np.concatenate([np.flatnonzero(src==s)[rng.integers(int((src==s).sum()),size=(10000,int((src==s).sum())))] for s in SOURCES],axis=1)
    data={k:np.array([q[k] for q in rows]) for k in rows[0] if k not in ['source','patient','channels','class_counts']}
    total=family2=family11=0
    def coord(v,row):return v.mean(-1) if row['coordinate']=='ALL12' else v[...,COORDS.index(row['coordinate'])]
    def contrast(name):
        if name.startswith('E1_H_'):return data['e1_h'][:,ARMS.index(name[5:])].mean(-1)
        if name.startswith('COMPOSITION_'):return data['composition_slope'][:,[25,50,75].index(int(name.split('_')[-1]))]
        f=[25,50,75].index(int(name.split('_')[-3]))
        if 'ABSGAP' in name:z=(data['e2_abs_class'][:,:,0]-data['e2_abs_class'][:,:,1]).mean(-1)
        else:z=data['e2_all'][:,:,4].mean(-1)
        return z[:,f]-z[:,3]
    for name,table in tables.items():
        if name in ['H_PERMUTATION','H_PERMUTATION_SUMMARY']:continue
        for row in table:
            ix=np.arange(len(rows)) if row['scope']=='ALL' else np.flatnonzero(src==row['scope'])
            if name in ['PRIMARY','SECONDARY']:values=contrast(row['endpoint']);fam=2 if name=='PRIMARY' else 11
            else:
                fam=0;arm=ARMS.index(row['arm']) if 'arm' in row else None;f=[.25,.5,.75,1.].index(row['fraction']) if 'fraction' in row else None
                if name=='E1_CLASS_LOSS':values=coord(data['e1_class'][:,['EZ','NEZ'].index(row['group']),arm],row)
                elif name=='E1_GAIN':
                    c=data['e1_class'][:,['EZ','NEZ'].index(row['group'])];values=coord(c[:,0]-c[:,arm],row)
                elif name=='E1_H':values=coord(data['e1_h'][:,arm],row)
                elif name=='E2_H':values=coord(data['e2_h'][:,f,arm],row)
                elif name in ['E2_LOSS','E2_GAIN']:
                    z=data['e2_all'][:,f] if row['group']=='ALL' else data['e2_class'][:,f,['EZ','NEZ'].index(row['group'])];values=coord(z[:,arm] if name=='E2_LOSS' else z[:,0]-z[:,arm],row)
                elif name=='E2_ANOMALY':
                    z=data['e2_abs_class'][:,f];e=row['endpoint'];values=coord(data['e2_abs_all'][:,f] if e=='ALL' else z[:,0]-z[:,1] if e=='gap' else z[:,['EZ','NEZ'].index(e)],row)
                elif name=='E2_SUPPORT':values=coord(data[row['endpoint']][:,f],row)
                elif name=='REFERENCE_COMPOSITION':values=data[row['endpoint']][:,f]
                elif name=='COMPOSITION_SLOPE':values=data['composition_slope'][:,f] if f<3 else np.full(len(rows),np.nan)
                else:raise AssertionError(name)
            values=values[ix];valid=np.isfinite(values);assert int(valid.sum())==row['patients']
            if valid.sum()<2:assert row['mean']=='';continue
            vv=np.full(len(rows),np.nan);vv[ix]=values;chosen=vv[draw];den=np.isfinite(chosen).sum(1);sample=np.nansum(chosen,axis=1)[den>0]/den[den>0]
            np.testing.assert_allclose([values[valid].mean(),np.median(values[valid]),np.quantile(sample,.025),np.quantile(sample,.975)],[row[k] for k in ['mean','median','ci_low','ci_high']],rtol=1e-9,atol=1e-10)
            if fam:
                np.testing.assert_allclose(np.quantile(sample,[.05/(2*fam),1-.05/(2*fam)]),[row['family_low'],row['family_high']],rtol=1e-9,atol=1e-10)
                if fam==2:family2+=1
                else:family11+=1
            total+=1
    for row in tables['H_PERMUTATION']:
        ix=np.arange(len(rows)) if row['scope']=='ALL' else np.flatnonzero(src==row['scope'])
        np.testing.assert_allclose(data['e1_null'][ix,row['permutation'],ARMS.index(row['arm'])].mean(),row['H'],rtol=1e-9,atol=1e-10)
    for row in tables['H_PERMUTATION_SUMMARY']:
        ix=np.arange(len(rows)) if row['scope']=='ALL' else np.flatnonzero(src==row['scope']);arm=ARMS.index(row['arm'])
        null=data['e1_null'][ix,:,arm].mean(0);obs=data['e1_h'][ix,arm].mean()
        tie=1e-10+1e-9*abs(obs)
        expected=[obs,null.mean(),np.quantile(null,.025),np.quantile(null,.975),(1+(null>=obs-tie).sum())/1001,(1+(null<=obs+tie).sum())/1001]
        np.testing.assert_allclose(expected,[row[k] for k in ['observed_H','null_mean','null_low','null_high','conditional_plus_one_upper','conditional_plus_one_lower']],rtol=1e-9,atol=1e-10)
    return dict(interval_rows=total,primary_family_intervals=family2,secondary_family_intervals=family11,permutation_values=len(tables['H_PERMUTATION']))

def verify(root):
    io.frozen(root);refs,old,labels,_=io.inputs();seal=io.read(root/'public/SEAL.json');assert io.sha(root/'private/ENDPOINTS.pkl')==seal['sha256']
    rows=io.load(root/'private/ENDPOINTS.pkl');verified=[];values=0;error=0.
    for i,(p,r) in enumerate(zip(refs,rows)):
        assert p['patient']==r['patient'];q=replay(p,labels[p['patient']],io.load(root/'private'/f'reference_{i:03d}.pkl'))
        for k,v in q.items():
            np.testing.assert_allclose(v,r[k],rtol=1e-9,atol=1e-10,equal_nan=True,err_msg=k);finite=np.isfinite(v);values+=v.size
            if finite.any():error=max(error,float(np.max(np.abs(v[finite]-r[k][finite]))))
        verified.append(dict(q,source=p['source'],patient=p['patient'],channels=r['channels'],class_counts=r['class_counts']))
        if (i+1)%10==0:print('INDEPENDENT_PATIENTS',i+1,flush=True)
    import stats_report
    tables=stats_report.summarize(rows);checks=independent_table_check(verified,tables);assert checks['primary_family_intervals']==2
    assert checks['secondary_family_intervals']==sum(r['status']=='estimable' for r in tables['SECONDARY'])
    io.write(root/'public/INDEPENDENT_VERIFICATION.json',dict(status='PASS',patients=80,channels=7635,endpoint_values_checked=values,max_absolute_error=error,absolute_tolerance=1e-10,relative_tolerance=1e-9,aggregate_checks=checks,secondary_slots=11,source_commit=io.commit(),frozen_sources_unchanged=True))
    print('INDEPENDENT_VERIFICATION_PASS',flush=True)
