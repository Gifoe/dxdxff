"""Release-only checks; never imported by training or permitted to change scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);args=parser.parse_args()
    admission=json.loads((HERE/'public_admission/ADMISSION.json').read_text(encoding='utf-8'))
    assert admission['status']=='PASS' and admission['expected_fresh_fits']==235 and admission['expected_reused_cells']==50
    for name,h in admission['source'].items(): assert sha(HERE/name)==h,(name,'SOURCE_CHANGED_AFTER_ADMISSION')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'])
    assert tests.returncode==0
    if args.package:
        namespace={'__name__':'release_allowlist_only'}
        import ast
        tree=ast.parse((HERE/'package_public.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets))
        allow=ast.literal_eval(node.value)
        with zipfile.ZipFile(args.package) as z:
            assert set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
            assert not manifest['private_artifacts_exported'] and set(manifest['files'])==set(allow)
            for n,h in manifest['files'].items():
                assert hashlib.sha256(z.read(n)).hexdigest()==h and z.getinfo(n).file_size<5000000
            independent=json.loads(z.read('INDEPENDENT_OUTER_VERIFICATION.json'))
            assert independent['status']=='PASS' and independent['patient_rows']==4560 and independent['patient_metric_checks']==41040
            assert independent['nominal_interval_checks']==351 and independent['primary_family_interval_checks']==36
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_CFA3_19_GROUP_3SEED_FIVE_FOLD_OUTER'
            counts={'MAIN_RESULTS.csv':19,'SEED_RESULTS.csv':57,'SEED_SD.csv':19,'FOLD_RESULTS.csv':285,
                'FOLD_SD.csv':57,'METRIC_COVERAGE.csv':513,'TRAIN_CELLS.csv':285,'RAW_COVERAGE.csv':285,
                'INTERVALS.csv':351,'PRIMARY_FAMILY_INTERVALS.csv':36,'FOLD_SEED_DIRECTIONS.csv':270,'COMPONENT_EVIDENCE.csv':18}
            forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
            for n in allow:
                content=z.read(n).decode('utf-8')
                assert not any(word in content for word in ['password=','PRIVATE KEY','fit_patient_ids'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(content.splitlines()));assert rows and not set(rows[0])&forbidden
                    if n in counts: assert len(rows)==counts[n],n
            for name,h in json.loads(z.read('ADMISSION.json'))['source'].items(): assert sha(HERE/name)==h
        print('PUBLIC_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(args.package),flush=True)
    print('IMMUTABLE17_SOURCE_AND_EXECUTABLE_TESTS_PASS',flush=True)


if __name__=='__main__':main()
