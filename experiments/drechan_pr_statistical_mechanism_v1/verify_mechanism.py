"""Second numerical implementation: reference arithmetic and bootstrap counts."""
from pathlib import Path
import numpy as np
import pandas as pd
import trusted_io as io

ARMS=['G0_GLOBAL','G1_CENTER','G2_SCALE','G3_FULL','DONOR_FULL','OP_GLOBAL','OP_PR']


def class_moments(x,y):
    origin=x[0].copy();shift=x-origin
    one=shift[y==1];zero=shift[y==0]
    u=np.sum(one,axis=0)/len(one);w=np.sum(zero,axis=0)/len(zero)
    e=one-u;n=zero-w
    covariance=np.zeros((x.shape[1],x.shape[1]))
    for j in range(x.shape[1]):
        for k in range(x.shape[1]):covariance[j,k]=.5*(np.dot(e[:,j],e[:,k])/len(one)+np.dot(n[:,j],n[:,k])/len(zero))
    return u-w,origin+.5*(u+w),covariance


def independent_bank(d,saved):
    raw=d['raw'].copy();tr=d['train'];pid=d['patient'];x=np.empty_like(raw)
    for j in range(12):
        fit=raw[tr,j];valid=np.isfinite(fit);fill=float(fit[valid].mean()) if valid.any() else 0.
        col=np.where(np.isnan(raw[:,j]),fill,raw[:,j]);mean=float(col[tr].sum()/len(tr))
        variance=float(np.square(col[tr]-mean).sum()/len(tr))
        unit=np.sqrt(variance) if variance>0 and np.ptp(col[tr])>0 else 1.
        x[:,j]=(col-mean)/unit
    reps={};stats={}
    for p in sorted(set(pid)):
        q=x[pid==p];mu=q.sum(0)/len(q);sd=np.sqrt(np.square(q-mu).sum(0)/len(q))
        stats[p]=(mu,np.where(sd>1e-8,sd,1.))
    op,_=io.reference('unit_pr').prepare(d['raw14'],pid,tr)
    for p in sorted(set(pid)):
        q=x[pid==p];mu,sd=stats[p];dm,ds=stats[saved['donors'][p]]
        reps[p]={'G0_GLOBAL':q,'G1_CENTER':q-mu,'G2_SCALE':q/sd,'G3_FULL':(q-mu)/sd,
            'DONOR_FULL':(q-dm)/ds,'OP_GLOBAL':q.astype(np.float32).astype(float),'OP_PR':op[pid==p][:,io.KEEP].astype(float)}
        for arm in ARMS:np.testing.assert_allclose(reps[p][arm],saved['reps'][p][arm],atol=1e-8,rtol=1e-8)
    return reps


def permute(y,pid,f,b,assessment):
    rng=np.random.default_rng([42,20261026,f,b,int(assessment)]);out=y.copy()
    for p in sorted(set(pid)):
        ix=np.flatnonzero(pid==p);order=rng.permutation(len(ix));out[ix]=y[ix[order]]
        assert out[ix].sum()==y[ix].sum()
    return out


def independent_model(reps,y,patients,arm,estimator):
    ms=[class_moments(reps[p][arm],y[p]) for p in sorted(patients)]
    delta=sum(q[0] for q in ms)/len(ms);cov=sum(q[2] for q in ms)/len(ms)
    lam=max(float(np.trace(cov))*.1/12,1e-12)
    a=delta if estimator=='DIRECT' else np.linalg.solve(cov+np.diag(np.full(12,lam)),delta)
    norm=np.sqrt(np.dot(a,a));v=a/norm if norm>1e-12 else np.zeros(12)
    noise=max(float(np.dot(v,np.dot(cov,v))),0.)
    reference=sum(float(q[1]@v) for q in ms)/len(ms)
    return dict(v=v,noise=noise,reference=reference,degenerate=bool(norm<=1e-12))


def endpoint(x,y,q,factor):
    # Direct projected scalar moments, not the primary covariance quadratic form.
    origin=x[0];shift=(x-origin)@q['v'];pos=shift[y==1];neg=shift[y==0]
    numerator=float(pos.mean()-neg.mean());variance=.5*(np.mean((pos-pos.mean())**2)+np.mean((neg-neg.mean())**2))
    delta,_,_=class_moments(x,y);norm=np.sqrt(delta@delta)
    A=float(delta@q['v']/norm) if norm>1e-12 else 0.
    eps=factor*max(q['noise'],1e-12);T=numerator/np.sqrt(variance+eps)
    midpoint=float(origin@q['v']+.5*(pos.mean()+neg.mean()));z=(midpoint-q['reference'])/np.sqrt(q['noise']+eps)
    return dict(T=T,A=A,z=z,MAE=abs(z))


COUNT={}


def draws(n):
    if n not in COUNT:
        ix=np.random.default_rng(42).integers(n,size=(10000,n));a=np.zeros((10000,n))
        np.add.at(a,(np.arange(10000)[:,None],ix),1);COUNT[n]=a
    return COUNT[n]


def boot(v,sd=False):
    v=np.asarray(v,float);count=draws(len(v));means=count@v/len(v)
    if not sd:return means
    return np.sqrt(np.maximum((count@(v*v)-len(v)*means*means)/(len(v)-1),0))


def assert_interval(samples,row,prefix='',adjust=False):
    alpha=.05/25 if adjust else .05
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2])
    keys=['adjusted_ci_low','adjusted_ci_high'] if adjust else [prefix+'ci_low',prefix+'ci_high']
    np.testing.assert_allclose([lo,hi],[row[k] for k in keys],atol=1e-7,rtol=1e-8)


def verify_tables(root,grouped,strict):
    pub=root/'public';original=grouped[grouped.permutation==-1];checked=0;adjusted=0
    for filename,metric in [('representation_alignment','A'),('transferable_snr','T'),('boundary_drift','MAE')]:
        for r in pd.read_csv(pub/(filename+'.csv'),float_precision='round_trip').to_dict('records'):
            v=original[(original.estimator==r['estimator'])&(original.factor==r['factor'])&(original.arm==r['arm'])].sort_values('patient')
            assert len(v)==80
            np.testing.assert_allclose([v[metric].mean(),v[metric].median()],[r['mean'],r['median']],atol=1e-7,rtol=1e-8)
            assert_interval(boot(v[metric]),r);checked+=1
            if metric=='MAE':
                np.testing.assert_allclose(v.z.std(ddof=1),r['SD'],atol=1e-7,rtol=1e-8)
                assert_interval(boot(v.z,True),r,'SD_');checked+=1
    for r in pd.read_csv(pub/'source_results.csv',float_precision='round_trip').to_dict('records'):
        v=original[(original.estimator==r['estimator'])&(original.factor==r['factor'])&(original.arm==r['arm'])&(original.source==r['source'])].sort_values('patient')
        assert len(v)==r['patients']==dict(hup=36,lzu=21,multicenter=15,pediatric=8)[r['source']]
        for metric in ['T','A','MAE']:
            np.testing.assert_allclose(v[metric].mean(),r[metric+'_mean'],atol=1e-7,rtol=1e-8)
            assert_interval(boot(v[metric]),r,metric+'_');checked+=1
        assert_interval(boot(v.z,True),r,'SD_');checked+=1
    for r in pd.read_csv(pub/'paired_intervals.csv',float_precision='round_trip').to_dict('records'):
        p=original[(original.estimator==r['estimator'])&(original.factor==r['factor'])]
        key='z' if r['metric']=='SD' else r['metric']
        a=p[p.arm==r['target']].sort_values('patient')[key].to_numpy();b=p[p.arm==r['control']].sort_values('patient')[key].to_numpy()
        if r['metric']=='SD':samples=boot(b,True)-boot(a,True);point=b.std(ddof=1)-a.std(ddof=1)
        else:
            delta=(b-a) if r['metric']=='MAE' else (a-b);samples=boot(delta);point=delta.mean()
            np.testing.assert_allclose(np.median(delta),r['median_paired'],atol=1e-7,rtol=1e-8)
            assert abs(float(np.mean(delta>1e-10))-r['positive_fraction'])<1e-12
        np.testing.assert_allclose(point,r['effect'],atol=1e-7,rtol=1e-8);assert_interval(samples,r);checked+=1
        if r['family_size']==25:assert_interval(samples,r,adjust=True);adjusted+=1
    for r in pd.read_csv(pub/'factorial_contrasts.csv',float_precision='round_trip').to_dict('records'):
        p=original[(original.estimator==r['estimator'])&(original.factor==r['factor'])]
        arrays=[p[p.arm==a].sort_values('patient')['z' if r['metric']=='SD' else r['metric']].to_numpy() for a in ARMS[:4]]
        coefficients={'CENTER':[-.5,.5,-.5,.5],'SCALE':[-.5,-.5,.5,.5],'INTERACTION':[1,-1,-1,1]}[r['component']]
        sgn=-1 if r['metric'] in ['SD','MAE'] else 1
        samples=sgn*sum(w*boot(v,r['metric']=='SD') for w,v in zip(coefficients,arrays))
        point=sgn*sum(w*(v.std(ddof=1) if r['metric']=='SD' else v.mean()) for w,v in zip(coefficients,arrays))
        np.testing.assert_allclose(point,r['effect'],atol=1e-7,rtol=1e-8);assert_interval(samples,r);checked+=1
        if r['family_size']==25:assert_interval(samples,r,adjust=True);adjusted+=1
    null=grouped[grouped.permutation>=0];nt=pd.read_csv(pub/'negative_controls.csv',float_precision='round_trip')
    for r in nt.to_dict('records'):
        v=null[(null.estimator==r['estimator'])&(null.permutation==r['permutation'])&(null.diagnostic==r['diagnostic'])&(null.arm==r['arm'])]
        assert len(v)==80
        actual=[v['T'].mean(),v.A.mean(),v.MAE.mean(),v.z.std(ddof=1)]
        np.testing.assert_allclose(actual,[r[k] for k in ['T','A','MAE','SD']],atol=1e-7,rtol=1e-8)
    pairs=pd.read_csv(pub/'paired_intervals.csv',float_precision='round_trip')
    for r in pd.read_csv(pub/'negative_control_summary.csv',float_precision='round_trip').to_dict('records'):
        sub=nt[(nt.estimator==r['estimator'])&(nt.diagnostic==r['diagnostic'])]
        a=sub[sub.arm==r['target']].sort_values('permutation')[r['metric']].to_numpy()
        if r['control']!='ABSOLUTE':
            b=sub[sub.arm==r['control']].sort_values('permutation')[r['metric']].to_numpy()
            a=b-a if r['metric'] in ['MAE','SD'] else a-b
        assert len(a)==100
        np.testing.assert_allclose([a.mean(),a.std(ddof=1),*np.quantile(a,[.025,.975])],
            [r[k] for k in ['null_mean','null_sample_SD','null_reference_low','null_reference_high']],atol=1e-7,rtol=1e-8)
    for r in pd.read_csv(pub/'strict_fit_sensitivity.csv',float_precision='round_trip').to_dict('records'):
        key='z' if r['metric']=='SD' else r['metric'];is_sd=r['metric']=='SD'
        a=strict[strict.arm==r['target']].sort_values('patient')[key].to_numpy();assert len(a)==79
        point=a.std(ddof=1) if is_sd else a.mean();samples=boot(a,is_sd)
        if r['control']!='ABSOLUTE':
            b=strict[strict.arm==r['control']].sort_values('patient')[key].to_numpy()
            if is_sd:point=b.std(ddof=1)-point;samples=boot(b,True)-samples
            elif r['metric']=='MAE':point=b.mean()-point;samples=boot(b-a)
            else:point-=b.mean();samples=boot(a-b)
        np.testing.assert_allclose(point,r['effect'],atol=1e-7,rtol=1e-8)
        assert_interval(samples,r);checked+=1
    assert checked==1112 and adjusted==25
    return checked,adjusted


def main(root):
    root=Path(root);data,audit=io.load_inputs();fr=io.jr(root/'private/FROZEN_BEFORE_ASSESS.json')
    assert fr['source']==io.sources() and io.sha(root/'private/DIRECTIONS.pkl')==fr['directions']
    seal=io.jr(root/'private/ASSESS_SEAL.json');assert seal['endpoint_sha256']==io.sha(root/'private/ASSESS_STATISTICS.pkl')
    bank=io.read(root/'private/PREPARED.pkl')['banks'];states=io.read(root/'private/DIRECTIONS.pkl')['states']
    private=io.read(root/'private/ASSESS_STATISTICS.pkl');original=pd.DataFrame(private['rows'])
    rows=[];models_checked=0
    for d,saved in zip(data,bank):
        reps=independent_bank(d,saved);pid=d['patient'];patients=sorted(d['roles']['assessment'])
        for state in [s for s in states if s['fold']==d['fold']]:
            y=d['y'].copy();b=state['permutation'];arm=state['arm'];est=state['estimator']
            if b>=0:
                ii=d['train'];y[ii]=permute(y[ii],pid[ii],d['fold'],b,False)
            yd={p:y[pid==p] for p in sorted(set(pid))}
            q=independent_model(reps,yd,d['roles']['train'],arm,est)
            for key in ['v','noise','reference']:np.testing.assert_allclose(q[key],state['model'][key],atol=1e-8,rtol=1e-8)
            models_checked+=1
            py=None
            if b>=0 and arm in ARMS[:4]:
                yy=d['y'].copy();ii=d['assessment'];yy[ii]=permute(yy[ii],pid[ii],d['fold'],b,True);py={p:yy[pid==p] for p in patients}
            for p in patients:
                for factor in [1e-8,1e-6,1e-10] if b==-1 else [1e-8]:
                    rows.append(dict(patient=p,source=saved['sources'][p],fold=d['fold'],estimator=est,arm=arm,factor=factor,
                        permutation=b,diagnostic='TRUE_ASSESS',strict_original_FIT=p in set(d['old']['patient']),
                        **endpoint(reps[p][arm],d['y'][pid==p],q,factor)))
                if py is not None:
                    rows.append(dict(patient=p,source=saved['sources'][p],fold=d['fold'],estimator=est,arm=arm,factor=1e-8,
                        permutation=b,diagnostic='MATCHED_H3_LABEL_NULL',strict_original_FIT=p in set(d['old']['patient']),
                        **endpoint(reps[p][arm],py[p],q,1e-8)))
        print('INDEPENDENT_PATH_REPLAYED',d['fold'],models_checked,len(rows),flush=True)
    actual=pd.DataFrame(rows);keys=['fold','patient','arm','estimator','factor','permutation','diagnostic']
    a=actual.set_index(keys).sort_index();b=original.set_index(keys).sort_index();assert a.index.equals(b.index)
    np.testing.assert_allclose(a[['T','A','z','MAE']],b[['T','A','z','MAE']],atol=1e-7,rtol=1e-8)
    err=np.max(np.abs(a[['T','A','z','MAE']].to_numpy()-b[['T','A','z','MAE']].to_numpy()))
    gkeys=['estimator','factor','permutation','diagnostic','arm','patient','source']
    grouped=actual.groupby(gkeys,sort=True)[['T','A','z','MAE']].mean().reset_index()
    strict=actual[(actual.permutation==-1)&(actual.estimator=='DIRECT')&(actual.factor==1e-8)&actual.strict_original_FIT]
    checked,adjusted=verify_tables(root,grouped,strict)
    assert models_checked==7070 and len(actual)==179360 and audit['ASSESS_unique']==80
    proof=dict(status='PASS',unique_assessed_patients=80,assessment_appearances=80,directions_reconstructed=7070,
        patient_direction_rows=179360,endpoint_values=717440,nominal_intervals_verified=checked,secondary_family_intervals=adjusted,
        strict_original_FIT_patients=79,original_validation_only_assessed=1,strict79_sensitivity_intervals=32,
        maximum_endpoint_absolute_discrepancy=float(err),tolerances=dict(atol=1e-7,rtol=1e-8),
        numerical_implementation='Independent origin-shifted class arithmetic,scalar projected variances,TRAIN covariance reconstruction andbootstrap count matrices',
        input_hashes_unchanged=True,private_exported=False,classifier_fits=0,neural_fits=0)
    io.write(root/'public/INDEPENDENT_VERIFICATION.json',proof)
    io.write(root/'public/STATUS.json',dict(status='COMPLETE_VERIFIED_PR_STATISTICAL_MECHANISM_80',patients=80,
        channels=7635,EZ=1743,NEZ=5892,ASSESS_appearances=80,primary_T_comparison=1,secondary_family=25,
        permutations=100,historical_exploratory=True,no_Outer_promotion=True,private_exported=False))
    print('INDEPENDENT_STATISTICAL_MECHANISM_PASS',proof,flush=True)
