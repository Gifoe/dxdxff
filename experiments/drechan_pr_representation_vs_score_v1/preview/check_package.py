"""Verify the exact anonymous preview allowlist BEFORE extraction."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path

FILES=['MAIN_RESULTS.csv','SOURCE_RESULTS.csv','SOURCE_SUMMARY.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','CPU_VALIDATION_REPLAY.json','INDEPENDENT_VERIFICATION.json','OUTER_STATUS.json']

def check(path):
    with zipfile.ZipFile(path) as z:
        names={'public/'+n for n in FILES}|{'PUBLIC_MANIFEST.json'}
        assert len(z.namelist())==len(names) and set(z.namelist())==names
        manifest=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(manifest['files'])==set(FILES) and manifest['private_data_included'] is False
        for n,s in manifest['files'].items():
            b=z.read('public/'+n);assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
            if n.endswith('.csv'):
                reader=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
                assert not set(reader.fieldnames)&{'patient','pid','channel','probability','prediction','embedding','logit','identity','raw'}
        iv=json.loads(z.read('public/INDEPENDENT_VERIFICATION.json'))
        assert iv['status']=='PASS' and iv['logical_cells']==70 and iv['patient_condition_rows']==1120 and iv['endpoint_values']==10080
        assert iv['nominal_intervals']==270 and iv['reported_primary_intervals']==4 and iv['planned_primary_family_size']==6
        assert iv['inference_device']=='CPU_FP32'
        assert json.loads(z.read('public/OUTER_STATUS.json'))['status']=='VERIFIED_EARLY_TWO_MODEL_OUTER_PREVIEW'
        replay=json.loads(z.read('public/CPU_VALIDATION_REPLAY.json'))
        assert replay['status']=='PASS' and len(replay['cells'])==20
        assert all(r['maximum_logit_error']<=1e-5 and r['maximum_probability_error']<=2e-6 and r['validation_classifications_exact'] for r in replay['cells'])
        def rows(n):return list(csv.DictReader(io.StringIO(z.read('public/'+n+'.csv').decode('utf-8-sig'))))
        main=rows('MAIN_RESULTS');assert len(main)==14 and all(int(r['patients'])==80 and int(r['channels'])==7635 for r in main)
        assert len(rows('SOURCE_RESULTS'))==56 and len(rows('SOURCE_SUMMARY'))==14 and len(rows('INTERVALS'))==270 and len(rows('PRIMARY_FAMILY_INTERVALS'))==4
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--extract',type=Path);a=p.parse_args()
    manifest=check(a.package)
    print('PREVIEW_PUBLIC_PACKAGE_CHECK_PASS',hashlib.sha256(a.package.read_bytes()).hexdigest())
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        with zipfile.ZipFile(a.package) as z:
            for n in FILES:(a.extract/n).write_bytes(z.read('public/'+n))
            (a.extract/'PUBLIC_MANIFEST.json').write_bytes(z.read('PUBLIC_MANIFEST.json'))
