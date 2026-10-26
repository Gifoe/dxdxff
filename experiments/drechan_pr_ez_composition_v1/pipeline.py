"""Immutable actual-data stages; no endpoint calculation before reference freeze."""
import argparse
from pathlib import Path
import shutil
import zipfile
import numpy as np
import core
import io_contract as io

def admit(root):
    assert not root.exists(),'New attempt required;preserve earlier runtimes'
    refs,old,labels,binding=io.inputs()
    for p in refs:
        order=core.rankings(p['patient'],p['channels']);masks=np.zeros_like(p['masks'])
        np.put_along_axis(masks,order[:,:len(p['x'])//2],True,axis=1)
        assert np.array_equal(masks,p['masks'])
        y=labels[p['patient']];assert set(y)=={0,1}
        for rank in order:
            subsets=core.nested_indices(rank)
            for q in range(len(rank)):
                assert all(q not in a[q] for a in subsets)
                assert all(set(subsets[f][q])<=set(subsets[f+1][q]) for f in range(3))
    io.write(root/'private/LOCK.json',dict(sources=io.sources(),binding_sha256=io.digest(binding),source_commit=io.commit()))
    io.write(root/'public/COVERAGE.json',binding)
    io.write(root/'public/ADMISSION.json',dict(status='PASS',patients=80,channels=7635,parent_binding=True,all_original_partitions_exact=True,all_queries_excluded=True,nested_references=True,no_new_endpoints_computed=True,source_commit=io.commit(),sources=io.sources()))
    print('ADMISSION_PASS all80 exact_parent_and_partitions',flush=True)

def freeze(root):
    refs,_,_,_=io.checked(root);files={};values=0
    for i,p in enumerate(refs):
        fp=core.freeze_patient(p);name=f'reference_{i:03d}.pkl';io.save(root/'private'/name,fp);files[name]=io.sha(root/'private'/name);values+=fp['parameters'].size
        assert np.array_equal(fp['parameters'][:,3],np.broadcast_to(fp['parameters'][0,3],fp['parameters'][:,3].shape))
        if (i+1)%10==0:print('FROZEN_PATIENTS',i+1,flush=True)
    manifest=dict(files=files,parent_reference_sha256=io.protocol()['parent_reference_sha256'],patients=80)
    io.write(root/'private/REFERENCE_MANIFEST.json',manifest)
    io.write(root/'public/FREEZE.json',dict(status='ALL_REFERENCES_FROZEN_BEFORE_ENDPOINTS',patients=80,reference_manifest_sha256=io.digest(manifest),reference_parameter_values=values,source_commit=io.commit(),sources=io.sources()))

def evaluate(root):
    io.frozen(root);refs,old,labels,_=io.inputs();rows=[];maxerr=0.
    for i,(p,legacy) in enumerate(zip(refs,old)):
        f=io.load(root/'private'/f'reference_{i:03d}.pkl');y=labels[p['patient']];e=core.evaluate_patient(p,f,y)
        np.testing.assert_allclose(e['e1_all'],legacy['crps'],rtol=1e-9,atol=1e-10)
        counts=np.array([(y==1).sum(),(y==0).sum()]);weighted=(e['e1_class']*counts[:,None,None]).sum(0)/len(y)
        np.testing.assert_allclose(weighted,legacy['crps'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(e['legacy_abs_class'],np.stack([legacy['EZ'],legacy['NEZ']]),rtol=1e-9,atol=1e-10)
        np.testing.assert_array_equal(e['e2_all'][:,0],np.broadcast_to(e['e2_all'][0,0],e['e2_all'][:,0].shape))
        np.testing.assert_array_equal(e['e2_all'][:,1],np.broadcast_to(e['e2_all'][0,1],e['e2_all'][:,1].shape))
        maxerr=max(maxerr,float(np.max(np.abs(e['e1_all']-legacy['crps']))))
        rows.append(dict(e,patient=p['patient'],source=p['source'],channels=len(y),class_counts=counts))
        if (i+1)%10==0:print('EVALUATED_PATIENTS',i+1,flush=True)
    io.save(root/'private/ENDPOINTS.pkl',rows)
    io.write(root/'public/SEAL.json',dict(status='ALL_ENDPOINTS_SEALED_BEFORE_REPORT',patients=80,channels=7635,sha256=io.sha(root/'private/ENDPOINTS.pkl'),parent_reconstruction_max_error=maxerr,sources=io.sources()))

def report(root):
    io.checked(root);assert io.read(root/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    import stats_report
    tables=stats_report.summarize(io.load(root/'private/ENDPOINTS.pkl'));stats_report.export(root/'public',tables)
    source=io.parent_root()/'public/ANOMALY.csv';target=root/'public/LEGACY_ANOMALY.csv';assert not target.exists();shutil.copyfile(source,target)
    assert io.sha(target)==io.sha(io.PRREF/'results/ANOMALY.csv')
    io.write(root/'public/STATUS.json',dict(status='COMPLETE_VERIFIED_PR_EZ_COMPOSITION_80',patients=80,channels=7635,primary_family=2,secondary_family=11,row_counts={k:len(v) for k,v in tables.items()},legacy_anomaly_sha256=io.sha(target),source_commit=io.commit()))

def package(root):
    io.checked(root);assert io.read(root/'public/STATUS.json')['status']=='COMPLETE_VERIFIED_PR_EZ_COMPOSITION_80'
    assert {p.name for p in (root/'public').iterdir()}==set(io.PAYLOADS)
    manifest=dict(schema='PR_EZ_COMPOSITION_PUBLIC_V1',files={n:io.sha(root/'public'/n) for n in io.PAYLOADS},sources=io.sources(),source_commit=io.commit())
    io.write(root/'public/MANIFEST.json',manifest);path=root/'public_aggregate_export_v1.zip';assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in io.PAYLOADS+['MANIFEST.json']:z.write(root/'public'/n,n)
    print('PUBLIC_PACKAGE_SHA256',io.sha(path),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['admit','freeze','evaluate','verify','report','package']);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    if a.stage=='verify':
        import independent;independent.verify(a.runtime)
    else:globals()[a.stage](a.runtime)
if __name__=='__main__':main()
