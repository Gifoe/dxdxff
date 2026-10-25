"""Equal unique-patient inference and anonymous publication export only after seal."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import mechanism_math as m
import trusted_io as io

PAIRS=[('G3_FULL','G0_GLOBAL'),('G1_CENTER','G0_GLOBAL'),('G2_SCALE','G0_GLOBAL'),
       ('G3_FULL','G1_CENTER'),('G3_FULL','G2_SCALE'),('G3_FULL','DONOR_FULL')]
ALLOW=['representation_alignment.csv','transferable_snr.csv','boundary_drift.csv','factorial_contrasts.csv',
 'paired_intervals.csv','negative_controls.csv','negative_control_summary.csv','source_results.csv',
 'patient_distribution_bins.csv','support_and_donor_audit.csv','fold_coverage.csv','INPUT_AUDIT.json',
 'FREEZE_STATUS.json','ASSESS_STATUS.json','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','STATUS.json',
 'strict_fit_sensitivity.csv']


def csv(path,rows):
    path=Path(path);assert not path.exists(),path.name
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')


def sd_stat(values,ids):
    v=np.asarray(values,float);s=v[ids].std(1,ddof=1);lo,hi=np.quantile(s,[.025,.975])
    return dict(SD=float(v.std(ddof=1)),SD_ci_low=float(lo),SD_ci_high=float(hi))


def strict_sensitivity(frame):
    q=frame[(frame.permutation==-1)&(frame.estimator=='DIRECT')&(frame.factor==1e-8)&frame.strict_original_FIT]
    q=m.aggregate(q);rows=[];ids=m.boot_ids(79)
    for arm in m.ARMS:
        v=q[q.arm==arm].sort_values('patient');assert v.patient.nunique()==len(v)==79
        for metric in ['T','A','MAE','SD']:
            value=v['z' if metric=='SD' else metric].to_numpy()
            samples=value[ids].std(1,ddof=1) if metric=='SD' else value[ids].mean(1)
            lo,hi=np.quantile(samples,[.025,.975])
            rows.append(dict(analysis='STRICT_ORIGINAL_FIT_79_SENSITIVITY',target=arm,control='ABSOLUTE',
                metric=metric,patients=79,effect=float(value.std(ddof=1) if metric=='SD' else value.mean()),
                ci_low=float(lo),ci_high=float(hi)))
    a=q[q.arm=='G3_FULL'].sort_values('patient');b=q[q.arm=='G0_GLOBAL'].sort_values('patient')
    for metric in ['T','A','MAE','SD']:
        key='z' if metric=='SD' else metric
        rows.append(dict(analysis='STRICT_ORIGINAL_FIT_79_SENSITIVITY',target='G3_FULL',control='G0_GLOBAL',
            metric=metric,**m.contrast(a[key],b[key],metric,ids)))
    return rows


def factorial(part,metric,kind,ids,family):
    vals={a:part[part.arm==a].sort_values('patient')['z' if metric=='SD' else metric].to_numpy() for a in m.FACTOR}
    if metric=='SD':
        p={a:v.std(ddof=1) for a,v in vals.items()};bs={a:v[ids].std(1,ddof=1) for a,v in vals.items()}
    else:p={a:v.mean() for a,v in vals.items()};bs={a:v[ids].mean(1) for a,v in vals.items()}
    coeff={'CENTER':[ -.5,.5,-.5,.5],'SCALE':[-.5,-.5,.5,.5],'INTERACTION':[1.,-1.,-1.,1.]}[kind]
    sign=-1 if metric in ['MAE','SD'] else 1
    point=sign*sum(w*p[a] for w,a in zip(coeff,m.FACTOR));samples=sign*sum(w*bs[a] for w,a in zip(coeff,m.FACTOR))
    lo,hi=np.quantile(samples,[.025,.975]);al,ah=np.quantile(samples,[.025/25,1-.025/25])
    identity=metric in ['T','A'] and kind in ['CENTER','INTERACTION']
    if identity:assert max(abs(point),abs(lo),abs(hi))<1e-8
    return dict(effect=float(point),ci_low=float(lo),ci_high=float(hi),
        adjusted_ci_low=float(al) if family else np.nan,adjusted_ci_high=float(ah) if family else np.nan,
        family_size=25 if family else 0,mathematical_identity=identity,
        scaling_alias=metric in ['T','A'] and kind=='SCALE')


def build_tables(frame,support):
    grouped=m.aggregate(frame);original=grouped[grouped.permutation==-1];alignment=[];snr=[];boundary=[];sources=[];pairs=[];fac=[];bins=[]
    assert len(original)==80*7*2*3
    for (est,factor),part in original.groupby(['estimator','factor'],sort=True):
        ids=m.boot_ids(80)
        for arm in m.ARMS:
            v=part[part.arm==arm].sort_values('patient');assert len(v)==v.patient.nunique()==80
            common=dict(estimator=est,factor=factor,arm=arm)
            alignment.append(dict(common,**m.summary(v.A,ids)))
            snr.append(dict(common,**m.summary(v['T'],ids)))
            boundary.append(dict(common,**m.summary(v.MAE,ids),**sd_stat(v.z,ids),signed_midpoint_mean=float(v.z.mean())))
            for source,q in v.groupby('source',sort=True):
                ni=m.boot_ids(len(q));r=dict(common,source=source,patients=len(q))
                for metric in ['T','A','MAE']:
                    s=m.summary(q[metric],ni)
                    r.update({metric+'_'+k:s[k] for k in ['mean','median','ci_low','ci_high']})
                r.update(sd_stat(q.z,ni));sources.append(r)
            for metric,edges in [('T',[-10,-5,-2,-1,0,1,2,5,10]),('A',np.linspace(-1,1,11)),('MAE',[0,.25,.5,1,2,5,10,20,50,100])]:
                hist=np.histogram(v[metric],bins=edges)[0]
                for j,count in enumerate(hist):bins.append(dict(common,metric=metric,bin_low=edges[j],bin_high=edges[j+1],count=int(count),
                    underflow=int((v[metric]<edges[0]).sum()) if j==0 else 0,overflow=int((v[metric]>edges[-1]).sum()) if j==len(hist)-1 else 0))
        for a,b in PAIRS+[('OP_PR','OP_GLOBAL')]:
            va=part[part.arm==a].sort_values('patient');vb=part[part.arm==b].sort_values('patient')
            assert va.patient.tolist()==vb.patient.tolist()
            for metric in ['T','A','MAE','SD']:
                identity=metric in ['T','A'] and (a,b) in [('G1_CENTER','G0_GLOBAL'),('G3_FULL','G2_SCALE')]
                primary=(a,b,metric,est,factor)==('G3_FULL','G0_GLOBAL','T','DIRECT',1e-8)
                family=est=='DIRECT' and factor==1e-8 and not identity and not primary and a!='OP_PR'
                pairs.append(dict(estimator=est,factor=factor,target=a,control=b,metric=metric,primary=primary,
                    **m.contrast(va['z' if metric=='SD' else metric],vb['z' if metric=='SD' else metric],metric,ids,identity,family)))
        for kind in ['CENTER','SCALE','INTERACTION']:
            for metric in ['T','A','MAE','SD']:
                family=est=='DIRECT' and factor==1e-8 and metric in ['MAE','SD']
                fac.append(dict(estimator=est,factor=factor,component=kind,metric=metric,patients=80,
                    **factorial(part,metric,kind,ids,family)))
        if est=='DIRECT' and factor==1e-8:
            full=part[part.arm=='G3_FULL'].sort_values('patient');base=part[part.arm=='G0_GLOBAL'].sort_values('patient')
            dif=full['T'].to_numpy()-base['T'].to_numpy()
            edges=np.array([-10,-3,-2,-1,-.5,-.2,0,.2,.5,1,2,3,10]);hist=np.histogram(dif,edges)[0]
            for j,count in enumerate(hist):bins.append(dict(estimator=est,factor=factor,arm='G3_MINUS_G0',metric='PAIRED_T',bin_low=edges[j],bin_high=edges[j+1],count=int(count),
                underflow=int((dif<edges[0]).sum()) if j==0 else 0,overflow=int((dif>edges[-1]).sum()) if j==len(hist)-1 else 0))
    assert sum(r['family_size']==25 for r in pairs)+sum(r['family_size']==25 for r in fac)==25
    negative=[];null=grouped[grouped.permutation>=0]
    for (e,b,diagnostic,arm),v in null.groupby(['estimator','permutation','diagnostic','arm'],sort=True):
        assert len(v)==80
        negative.append(dict(estimator=e,permutation=int(b),diagnostic=diagnostic,arm=arm,patients=80,
            T=float(v['T'].mean()),A=float(v.A.mean()),MAE=float(v.MAE.mean()),SD=float(v.z.std(ddof=1))))
    nt=pd.DataFrame(negative);negative_summary=[]
    for (e,diagnostic),q in nt.groupby(['estimator','diagnostic']):
        for a,b in [('G3_FULL','G0_GLOBAL'),('G1_CENTER','G0_GLOBAL'),('G2_SCALE','G0_GLOBAL')]:
            qa=q[q.arm==a].set_index('permutation').sort_index();qb=q[q.arm==b].set_index('permutation').sort_index()
            if not len(qa):continue
            for metric in ['T','A','MAE','SD']:
                vals=(qb[metric]-qa[metric]) if metric in ['MAE','SD'] else (qa[metric]-qb[metric])
                obs=next(r['effect'] for r in pairs if r['estimator']==e and r['factor']==1e-8 and r['target']==a and r['control']==b and r['metric']==metric)
                negative_summary.append(dict(estimator=e,diagnostic=diagnostic,target=a,control=b,metric=metric,repetitions=100,
                    observed=obs,null_mean=float(vals.mean()),null_sample_SD=float(vals.std(ddof=1)),
                    null_reference_low=float(vals.quantile(.025)),null_reference_high=float(vals.quantile(.975))))
        for arm in sorted(set(q.arm)):
            for metric in ['T','A','MAE','SD']:
                vals=q[q.arm==arm][metric];v=original[(original.estimator==e)&(original.factor==1e-8)&(original.arm==arm)]
                observed=float(v.z.std(ddof=1)) if metric=='SD' else float(v[metric].mean())
                negative_summary.append(dict(estimator=e,diagnostic=diagnostic,target=arm,control='ABSOLUTE',metric=metric,repetitions=100,
                    observed=observed,null_mean=float(vals.mean()),null_sample_SD=float(vals.std(ddof=1)),
                    null_reference_low=float(vals.quantile(.025)),null_reference_high=float(vals.quantile(.975))))
    su=pd.DataFrame(support);donors=[]
    for f,v in su.groupby('fold'):
        donors.append(dict(fold=int(f),assessment_patients=len(v),supported_coordinates=int(v.supported_coordinates.sum()),
            donor_supported_coordinates=int(v.donor_supported_coordinates.sum()),same_source_pairs=int(v.donor_same_source.sum()),
            same_channel_bin_pairs=int(v.donor_same_count_bin.sum()),median_log2_channel_distance=float(v.donor_log2_count_distance.median()),
            median_effective_scale_ratio=float(v.donor_effective_scale_ratio_median.median()),
            mean_input_variance_correct=float(v.variance_correct.mean()),mean_input_variance_donor=float(v.variance_donor.mean())))
    return dict(representation_alignment=alignment,transferable_snr=snr,boundary_drift=boundary,factorial_contrasts=fac,
        paired_intervals=pairs,source_results=sources,patient_distribution_bins=bins,
        negative_controls=negative,negative_control_summary=negative_summary,support_and_donor_audit=donors),grouped


def report(root):
    root=Path(root);seal=io.jr(root/'private/ASSESS_SEAL.json');assert seal['endpoint_sha256']==io.sha(root/'private/ASSESS_STATISTICS.pkl')
    pack=io.read(root/'private/ASSESS_STATISTICS.pkl');frame=pd.DataFrame(pack['rows'])
    assert len(frame)==179360
    tables,grouped=build_tables(frame,pack['support'])
    tables['strict_fit_sensitivity']=strict_sensitivity(frame)
    io.save(root/'private/UNIQUE_PATIENT_STATISTICS.pkl',grouped)
    for name,rows in tables.items():csv(root/'public'/(name+'.csv'),rows)
    banks,_=io.load_inputs();coverage=[]
    for d in banks:
        ii=d['assessment'];coverage.append(dict(fold=d['fold'],TRAIN_patients=len(d['roles']['train']),CAL_patients=8,
            ASSESS_patients=16,original_validation_only_assessment=d['original_validation_only_assessment'],
            ASSESS_channels=len(ii),EZ_channels=int(d['y'][ii].sum()),NEZ_channels=int((1-d['y'][ii]).sum())))
    assert sum(r['ASSESS_channels'] for r in coverage)==7635 and sum(r['EZ_channels'] for r in coverage)==1743
    csv(root/'public/fold_coverage.csv',coverage)
    io.write(root/'public/REPORT_STATUS.json',dict(status='ALL_FROZEN_PROBES_REPORTED_PENDING_INDEPENDENT_CHECK',
        unique_patients=80,appearances=80,patient_direction_rows=len(frame),directions=7070,
        primary_comparisons=1,secondary_family_slots=25,TRAIN_label_permutations=100,neural_fits=0,classifier_fits=0,
        row_counts={k:len(v) for k,v in tables.items()}))
    print('REPORT_COMPLETE all80patients,7070directions,179360patient-direction rows,100nullrepeats',flush=True)


def package(root):
    root=Path(root);pub=root/'public';assert io.jr(pub/'STATUS.json')['status']=='COMPLETE_VERIFIED_PR_STATISTICAL_MECHANISM_80'
    manifest=dict(schema='PR_STATISTICAL_MECHANISM_V1',source=io.sources(),files={n:io.sha(pub/n) for n in ALLOW},private_exported=False)
    io.write(pub/'PACKAGE_MANIFEST.json',manifest);p=root/'public_aggregate_export_v1.zip';assert not p.exists()
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ALLOW+['PACKAGE_MANIFEST.json']:z.write(pub/n,n)
    print('ANONYMOUS_MECHANISM_PACKAGE',io.sha(p),p.stat().st_size,flush=True)
