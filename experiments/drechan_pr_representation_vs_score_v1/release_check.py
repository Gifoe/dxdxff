"""Exact public allowlist, whole-manifest verification before extraction."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path

FILES=['MAIN_RESULTS.csv','SOURCE_RESULTS.csv','SOURCE_SUMMARY.csv','FOLD_RESULTS.csv','COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','MODEL_SELECTION.csv','REUSE_AUDIT.csv','ADMISSION.json','SOURCE_LOCK_PUBLIC.json','FREEZE_STATUS.json','SCORE_STATUS.json','ENDPOINT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT_STATUS.json','OUTER_STATUS.json','FINDINGS.md']
TERMINAL='COMPLETE_VERIFIED_PR_REPRESENTATION_VS_SCORE_3_MODELS_7_CONDITIONS_5_FOLDS'

def check_package(path):
    with zipfile.ZipFile(path) as z:
        expected={'public/'+n for n in FILES}|{'PUBLIC_MANIFEST.json'};assert set(z.namelist())==expected and len(z.namelist())==len(expected)
        manifest=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(manifest['files'])=={'public/'+n for n in FILES}
        for name,spec in manifest['files'].items():
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==spec['sha256'] and len(raw)==spec['bytes']
            if name.endswith('.csv'):
                reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
                assert not set(reader.fieldnames)&{'patient','pid','channel','prediction','probability','embedding','logit','raw','identity'}
        terminal=json.loads(z.read('public/OUTER_STATUS.json'));verification=json.loads(z.read('public/INDEPENDENT_VERIFICATION.json'))
        assert terminal['status']==TERMINAL and verification['status']=='PASS'
        assert verification['logical_cells']==105 and verification['patient_condition_rows']==1680 and verification['endpoint_values']==15120
    return manifest

def build_package(runtime):
    runtime=Path(runtime);pub=runtime/'public';files={}
    for n in FILES:
        b=(pub/n).read_bytes();files['public/'+n]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    manifest=dict(schema='PR_REPRESENTATION_VS_SCORE_PUBLIC_V1',files=files,private_data_included=False)
    path=runtime/'public_aggregate_export_v1.zip';tmp=path.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in FILES:z.write(pub/n,'public/'+n)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2))
    check_package(tmp);tmp.replace(path);return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();m=check_package(a.package)
    print('PUBLIC_PACKAGE_CHECK_PASS',len(m['files']))
