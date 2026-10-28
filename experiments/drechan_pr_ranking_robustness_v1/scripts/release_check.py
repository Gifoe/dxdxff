"""Check the exact anonymous A-C archive before allowlist extraction."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
FILES=['ABC_VERIFICATION.json','COVERAGE_CURVE.csv','manifest.json','PATIENT_DISTRIBUTION_BINS.csv','STATUS.json','TABLE_A.csv','TABLE_B.csv','TABLE_B_SOURCE.csv','TABLE_C_SOURCE.csv','TABLE_C_SUMMARY.csv','TABLE_D_PAIRED.csv','TABLE_D_REVIEW.csv']
DFILES=['D_ADMISSION.json','D_VERIFICATION.json','TABLE_E.csv','TABLE_E_PAIRED.csv','TABLE_E_SOURCE.csv','PROVENANCE_SUPPLEMENT.json']
def check(path):
    with zipfile.ZipFile(path) as z:
        complete='public/D_VERIFICATION.json' in z.namelist();files=FILES+DFILES if complete else FILES
        expected={'public/'+n for n in files}|{'PUBLIC_MANIFEST.json'}
        assert set(z.namelist())==expected and len(z.namelist())==len(expected)
        man=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(man['files'])==set(files) and man['private_data_included'] is False
        for n,s in man['files'].items():
            b=z.read('public/'+n);assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
            if n.endswith('.csv'):
                r=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
                assert not set(r.fieldnames)&{'patient','pid','channel','prediction','probability','score','logit','raw','embedding','identity'}
        iv=json.loads(z.read('public/ABC_VERIFICATION.json'));assert iv['status']=='PASS' and iv['logical_cells']==45 and iv['patient_rows']==720
        assert iv['primary_family_intervals']==6 and iv['B_family_intervals']==12 and iv['C_family_intervals']==9 and iv['ranking_invariance']=='EXACT'
        status=json.loads(z.read('public/STATUS.json'))
        if complete:
            assert status['status']=='ABCD_COMPLETE_VERIFIED' and status['D']=='COMPLETE'
            d=json.loads(z.read('public/D_VERIFICATION.json'));a=json.loads(z.read('public/D_ADMISSION.json'))
            assert d['status']=='PASS' and d['patient_model_cells']==342 and d['patient_K_rows']==1026 and d['repetition_rows']==3666
            assert d['family_intervals']==36 and d['nominal_paired_intervals']==144 and d['all_cached_direct_representation_and_logit_bit_exact']
            assert a['included_patients']==57 and a['common_channels']==5735 and a['excluded_channels']==128
            assert json.loads(z.read('public/manifest.json'))['D_pending'] is False
            assert json.loads(z.read('public/PROVENANCE_SUPPLEMENT.json'))['D_scores_pending'] is False
        else:assert status['status']=='ABC_COMPLETE_VERIFIED_D_PENDING'
    return man
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--extract',type=Path);a=p.parse_args();m=check(a.package)
    print('ABC_PUBLIC_ARCHIVE_CHECK_PASS',hashlib.sha256(a.package.read_bytes()).hexdigest())
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        with zipfile.ZipFile(a.package) as z:
            for n in m['files']:(a.extract/n).write_bytes(z.read('public/'+n))
            (a.extract/'PUBLIC_MANIFEST.json').write_bytes(z.read('PUBLIC_MANIFEST.json'))
