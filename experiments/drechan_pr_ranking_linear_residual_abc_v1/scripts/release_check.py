"""Strict anonymous release validation, before any archive extraction."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
FILES=['ABC_VERIFICATION.json','ADMISSION.json','CONDITION_MAP.json','COVERAGE_CURVE.csv','manifest.json','PATIENT_DISTRIBUTION_BINS.csv','STATUS.json','TABLE_A.csv','TABLE_B.csv','TABLE_B_SOURCE.csv','TABLE_C_SOURCE.csv','TABLE_C_SUMMARY.csv','TABLE_D_PAIRED.csv','TABLE_D_REVIEW.csv']
def check(path):
    with zipfile.ZipFile(path) as z:
        expected={'public/'+n for n in FILES}|{'PUBLIC_MANIFEST.json'}
        assert len(z.namelist())==len(expected) and set(z.namelist())==expected
        man=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(man['files'])==set(FILES) and man['private_data_included'] is False
        for n,meta in man['files'].items():
            b=z.read('public/'+n);assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256']
            if n.endswith('.csv'):
                rr=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
                assert not set(rr.fieldnames)&{'patient','pid','channel','prediction','probability','score','logit','raw','embedding','identity'}
                for r in rr:
                    assert not (r.get('model')=='TIMECONV' and r.get('condition')=='FULL_PR')
        iv=json.loads(z.read('public/ABC_VERIFICATION.json'))
        assert iv['status']=='PASS' and iv['logical_cells']==45 and iv['patient_rows']==720 and iv['channels_per_condition']==7635
        assert iv['primary_family_intervals']==6 and iv['B_family_intervals']==12 and iv['C_family_intervals']==9
        assert iv['main_table_all11_metrics_match'] and iv['unchanged_patient_rows_exact']==640 and iv['frozen_parents_unchanged']
        a=json.loads(z.read('public/ADMISSION.json'));assert a['status']=='PASS' and a['cached_fixed_head_replays']==80 and a['new_fits']==0
        s=json.loads(z.read('public/STATUS.json'));assert s['status']=='ABC_PR_LINEAR_RESIDUAL_COMPLETE_VERIFIED' and s['C']=='COMPLETE'
        assert json.loads(z.read('public/manifest.json'))['timeconv_version']=='FROZEN_BACKBONE_PR_LINEAR_RESIDUAL'
    return man
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--extract',type=Path);a=p.parse_args();m=check(a.package)
    print('PR_LINEAR_RESIDUAL_ABC_PUBLIC_CHECK_PASS',hashlib.sha256(a.package.read_bytes()).hexdigest())
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        with zipfile.ZipFile(a.package) as z:
            for n in m['files']:(a.extract/n).write_bytes(z.read('public/'+n))
            (a.extract/'PUBLIC_MANIFEST.json').write_bytes(z.read('PUBLIC_MANIFEST.json'))
