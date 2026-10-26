"""admit -> freeze -> evaluate -> verify -> report -> package, immutable barriers."""
import argparse
import sys
import zipfile
from pathlib import Path
import numpy as np
import contract as io
import math_core as m

def admit(root):
    assert not root.exists(),'Use a new runtime; never overwrite attempts'
    data,binding=io.inputs();maximum=0.;sourcecounts=[]
    for d in data:
        assert set(d['roles']['train']).isdisjoint(d['roles']['assessment'])
        raw=np.asarray(d['raw14'],float);pid=d['patient'];tr=d['train'];unit=io.parent.reference('unit_pr');feature=io.parent.reference('feature_pr')
        op,pre=unit.prepare(raw,pid,tr);original,_=feature.prepare(raw,pid,tr);assert np.array_equal(op,original)
        # Independently rebuild the full14D,full-channel operational path.
        z=np.zeros_like(raw,dtype=float);units=pre['TRAIN_unit_scale']
        for p in sorted(set(pid)):
            ix=pid==p;mu=raw[ix].mean(0);sd=raw[ix].std(0,ddof=0)
            z[ix]=np.divide(raw[ix]-mu,sd,out=np.zeros_like(raw[ix],dtype=float),where=sd>1e-8*units)
        replay=((z-pre['mean'])/pre['scale']).astype(np.float32);assert np.array_equal(op,replay)
        pr=io.geometry.fit_global(d['raw'][tr]);x=io.geometry.global_query(d['raw'],pr)
        assert np.isfinite(x).all()
        for source in m.SOURCES:
            n=len(set(pid[tr][d['source'][tr]==source]));assert n>0;sourcecounts.append(n)
        for p in sorted(d['roles']['assessment']):
            ix=np.flatnonzero(pid==p);assert len(ix)>=4 and set(d['y'][ix])=={0,1}
            masks=m.split_masks(p,d['channel'][ix])
            assert masks.shape==(100,len(ix)) and (masks.sum(1)==len(ix)//2).all()
    binding.update(primary_patients=80,primary_channels=7635,reference_repeats=100)
    io.write(root/'private/LOCK.json',dict(sources=io.sources(),binding_sha256=io.digest(io.inputs()[1]),source_commit=io.commit()))
    io.write(root/'public/COVERAGE.json',binding)
    io.write(root/'public/ADMISSION.json',dict(status='PASS',full14_operational_parity='bit_exact',all80_channel_support=True,all_sources_in_each_TRAIN=True,source_TRAIN_patient_counts=sourcecounts,no_new_outcomes_computed=True,sources=io.sources(),source_commit=io.commit()))
    print('ADMISSION_PASS all80 exact_roles exact_operational_PR',flush=True)

def frozen_data(root):
    io.checked(root)
    f=io.read(root/'public/FREEZE.json');assert io.sha(root/'private/REFERENCES.pkl')==f['references_sha256']
    return io.load(root/'private/REFERENCES.pkl')

def freeze(root):
    data,binding=io.checked(root)
    patients=[]
    for d in data:
        pre=io.geometry.fit_global(d['raw'][d['train']]);x=io.geometry.global_query(d['raw'],pre)
        glob=m.moments(x[d['train']]);stats={s:m.moments(x[d['train']][d['source'][d['train']]==s]) for s in m.SOURCES}
        for p in sorted(d['roles']['assessment']):
            ix=np.flatnonzero(d['patient']==p);channels=d['channel'][ix].astype(str)
            masks=m.split_masks(p,channels);ref=m.reference_parameters(x[ix],masks);source=str(d['source'][ix[0]])
            patients.append(dict(patient=p,fold=d['fold'],source=source,archive_indices=d['archive_indices'][ix],channels=channels,x=x[ix],global_pre=pre,masks=masks,references=ref,**{'global':glob,'source_stats':stats[source]}))
    patients.sort(key=lambda p:(m.SOURCES.index(p['source']),p['patient']))
    assert len(patients)==80 and sum(len(p['x']) for p in patients)==7635 and len({p['patient'] for p in patients})==80
    io.save(root/'private/REFERENCES.pkl',patients)
    io.write(root/'public/FREEZE.json',dict(status='ALL_REFERENCES_FROZEN_BEFORE_EVALUATION',references_sha256=io.sha(root/'private/REFERENCES.pkl'),patients=80,partitions=8000,directional_reference_sets=16000,queried_channel_appearances=763500,source_commit=io.commit(),sources=io.sources()))
    print('FREEZE_PASS 8000_partitions 16000_references',flush=True)

def evaluate(root):
    patients=frozen_data(root);data,_=io.inputs();lookup={}
    for d in data:
        for p in d['roles']['assessment']:lookup[p]=d['y'][d['patient']==p]
    rows=[]
    for i,p in enumerate(patients):
        e=m.patient_endpoints(p);y=lookup[p['patient']]
        e.update(patient=p['patient'],source=p['source'],fold=p['fold'],channels=len(y),EZ=e['deviation'][y==1].mean(0),NEZ=e['deviation'][y==0].mean(0),permutation=m.permuted_gaps(e['deviation'],y,p['patient']))
        rows.append(e)
        if (i+1)%10==0:print('EVALUATED_PATIENTS',i+1,flush=True)
    io.save(root/'private/ENDPOINTS.pkl',rows)
    io.write(root/'public/SEAL.json',dict(status='ALL_ENDPOINTS_SEALED_BEFORE_REPORT',sha256=io.sha(root/'private/ENDPOINTS.pkl'),patients=80,crps_patient_arm_coordinate_values=4800,permutation_patient_values=80000,sources=io.sources()))

def package(root):
    assert io.read(root/'public/STATUS.json')['status']=='COMPLETE_VERIFIED_PR_REFERENCE_TRANSFER_80'
    assert {p.name for p in (root/'public').iterdir()}==set(io.PAYLOADS)
    manifest=dict(schema='PR_REFERENCE_TRANSFER_PUBLIC_V1',files={n:io.sha(root/'public'/n) for n in io.PAYLOADS},sources=io.sources(),source_commit=io.commit(),privacy='Aggregate only;no individuals or complementary subcohorts')
    io.write(root/'public/MANIFEST.json',manifest);path=root/'public_aggregate_export_v1.zip';assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name in io.PAYLOADS+['MANIFEST.json']:z.write(root/'public'/name,name)
    print('PUBLIC_PACKAGE_SHA256',io.sha(path),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['admit','freeze','evaluate','verify','report','package']);a.add_argument('--runtime',type=Path,required=True);args=a.parse_args();root=args.runtime
    if args.stage=='admit':admit(root)
    elif args.stage=='freeze':freeze(root)
    elif args.stage=='evaluate':evaluate(root)
    elif args.stage=='verify':
        import independent;independent.verify(root)
    elif args.stage=='report':
        io.checked(root)
        import stats_report;stats_report.report(root)
    else:io.checked(root);package(root)
if __name__=='__main__':main()
