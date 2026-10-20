"""Release-only immutable-source and anonymous package validation."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);a=p.parse_args()
    admission=HERE/'public_admission/ADMISSION.json'
    if admission.exists():
        proof=json.loads(admission.read_text());assert proof['status']=='PASS' and proof['expected_fresh_fits']==15 and proof['expected_reused_cells']==10
        for n,h in proof['source'].items():assert sha(HERE/n)==h,(n,'SOURCE_CHANGED_AFTER_ADMISSION')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert result.returncode==0
    if a.package:
        assert admission.exists()
        tree=ast.parse((HERE/'package_public.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets));allow=ast.literal_eval(node.value)
        with zipfile.ZipFile(a.package) as z:
            assert set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert not manifest['private_artifacts_exported'] and set(manifest['files'])==set(allow)
            for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h and z.getinfo(n).file_size<5000000
            verify=json.loads(z.read('INDEPENDENT_OUTER_VERIFICATION.json'))
            assert verify['status']=='PASS' and verify['patient_rows']==400 and verify['patient_metric_checks']==3600
            assert verify['nominal_interval_checks']==90 and verify['primary_family_interval_checks']==10
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_PCA_5_GROUP_SEED42_FIVE_FOLD_OUTER'
            counts={'MAIN_RESULTS.csv':5,'FOLD_RESULTS.csv':25,'FOLD_SD.csv':5,'METRIC_COVERAGE.csv':45,'TRAIN_CELLS.csv':25,'RAW_COVERAGE.csv':25,'INTERVALS.csv':90,'PRIMARY_FAMILY_INTERVALS.csv':10,'FOLD_DIRECTIONS.csv':25,'COMPONENT_EVIDENCE.csv':5}
            forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
            for n in allow:
                text=z.read(n).decode('utf-8');assert not any(s in text for s in ['password=','PRIVATE KEY','fit_patient_ids'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(text.splitlines()));assert rows and not set(rows[0])&forbidden
                    if n in counts:assert len(rows)==counts[n],n
            for n,h in json.loads(z.read('ADMISSION.json'))['source'].items():assert sha(HERE/n)==h
        print('ANONYMOUS_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(a.package))
    print('EXECUTABLE_TESTS_AND_AVAILABLE_SOURCE_LOCK_PASS')
if __name__=='__main__':main()
