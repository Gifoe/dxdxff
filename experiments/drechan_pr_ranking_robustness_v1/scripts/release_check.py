"""Check the exact anonymous A-C archive before allowlist extraction."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
FILES=['ABC_VERIFICATION.json','COVERAGE_CURVE.csv','manifest.json','PATIENT_DISTRIBUTION_BINS.csv','STATUS.json','TABLE_A.csv','TABLE_B.csv','TABLE_B_SOURCE.csv','TABLE_C_SOURCE.csv','TABLE_C_SUMMARY.csv','TABLE_D_PAIRED.csv','TABLE_D_REVIEW.csv']
def check(path):
    with zipfile.ZipFile(path) as z:
        expected={'public/'+n for n in FILES}|{'PUBLIC_MANIFEST.json'}
        assert set(z.namelist())==expected and len(z.namelist())==len(expected)
        man=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(man['files'])==set(FILES) and man['private_data_included'] is False
        for n,s in man['files'].items():
            b=z.read('public/'+n);assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
            if n.endswith('.csv'):
                r=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
                assert not set(r.fieldnames)&{'patient','pid','channel','prediction','probability','score','logit','raw','embedding','identity'}
        iv=json.loads(z.read('public/ABC_VERIFICATION.json'));assert iv['status']=='PASS' and iv['logical_cells']==45 and iv['patient_rows']==720
        assert iv['primary_family_intervals']==6 and iv['B_family_intervals']==12 and iv['C_family_intervals']==9 and iv['ranking_invariance']=='EXACT'
        assert json.loads(z.read('public/STATUS.json'))['status']=='ABC_COMPLETE_VERIFIED_D_PENDING'
    return man
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--extract',type=Path);a=p.parse_args();m=check(a.package)
    print('ABC_PUBLIC_ARCHIVE_CHECK_PASS',hashlib.sha256(a.package.read_bytes()).hexdigest())
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        with zipfile.ZipFile(a.package) as z:
            for n in FILES:(a.extract/n).write_bytes(z.read('public/'+n))
            (a.extract/'PUBLIC_MANIFEST.json').write_bytes(z.read('PUBLIC_MANIFEST.json'))
