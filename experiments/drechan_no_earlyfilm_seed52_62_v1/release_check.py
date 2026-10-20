"""Release-only CPU tests, immutable sources and exact anonymous schemas."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);a=parser.parse_args()
    proofpath=HERE/'public_admission/ADMISSION.json';proof=None
    if proofpath.exists():
        proof=json.loads(proofpath.read_text());assert proof['status']=='PASS'
        assert proof['expected_fresh_fits']==10 and proof['expected_reused_cells']==0 and proof['initialization_cells_checked']==10
        for n,h in proof['source'].items():assert sha(HERE/n)==h,(n,'SOURCE_CHANGED_AFTER_ADMISSION')
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert r.returncode==0
    for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
    if a.package:
        assert proof
        tree=ast.parse((HERE/'package_public.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets))
        allow=ast.literal_eval(node.value)
        with zipfile.ZipFile(a.package) as z:
            assert len(z.namelist())==29 and set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert not manifest['private_artifacts_exported'] and set(manifest['files'])==set(allow)
            for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h and z.getinfo(n).file_size<5000000
            verify=json.loads(z.read('INDEPENDENT_OUTER_VERIFICATION.json'))
            assert verify['status']=='PASS' and verify['patient_rows']==160 and verify['patient_metric_checks']==1440 and verify['nominal_interval_checks']==27
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_NO_EARLY_SEEDS52_62_FIVE_FOLD_OUTER'
            assert json.loads(z.read('ADMISSION.json'))==proof
            counts={'MAIN_RESULTS.csv':1,'SEED_RESULTS.csv':2,'SEED_SD.csv':1,'FOLD_RESULTS.csv':10,'FOLD_SD.csv':2,
                'METRIC_COVERAGE.csv':18,'TRAIN_CELLS.csv':10,'RAW_COVERAGE.csv':10,'INTERVALS.csv':27,
                'ROLE_AUDIT.csv':5,'INITIALIZATION_AUDIT.csv':10,'VALIDATION_REPLAY_CHECKS.csv':10,'PREPROCESSING_REPLAY.csv':5,
                'POOLED_RESULTS.csv':2,'POOLED_SEED_MEAN.csv':1}
            forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
            for n in allow:
                text=z.read(n).decode('utf-8');assert not any(s in text for s in ['password=','PRIVATE KEY','fit_patient_ids'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(text.splitlines()));assert rows and not set(rows[0])&forbidden
                    if n in counts:assert len(rows)==counts[n],n
            for n,h in proof['source'].items():assert sha(HERE/n)==h
        print('EXPLICIT_ANONYMOUS_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(a.package))
    print('EXECUTABLE_TESTS_AND_SOURCE_LOCK_PASS')
if __name__=='__main__':main()
