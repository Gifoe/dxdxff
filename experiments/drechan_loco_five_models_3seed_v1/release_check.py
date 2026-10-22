"""Local CPU/anonymous allowlist release verifier; never imports server data."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--admission',type=Path);parser.add_argument('--package',type=Path);args=parser.parse_args()
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert r.returncode==0
    proofroot=HERE/'public_admission'
    if args.admission:
        with zipfile.ZipFile(args.admission) as z:
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert manifest['schema']=='LOCO5_3SEED_ADMISSION_V1'
            expected={'REUSE_AUDIT.json'}|{f'seed{s}/{n}' for s in [42,43,44] for n in ['ADMISSION.json','ROLE_AUDIT.csv','INPUT_AUDIT.json','INITIALIZATION_AUDIT.csv']}
            assert set(z.namelist())==expected|{'PACKAGE_MANIFEST.json'} and set(manifest['files'])==expected
            assert manifest['private_artifacts_exported'] is False
            for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
            for s in [42,43,44]:
                q=json.loads(z.read(f'seed{s}/ADMISSION.json'))
                assert q['status']=='PASS' and q['initialization_cells']==16 and q['no_test_inference']
                assert q['source']==manifest['source'] and q['binding']['train_seed']==s and q['binding']['split_seed']==42
                for n,h in q['source'].items():assert sha(HERE/n)==h,(n,'SOURCE_CHANGED')
            assert json.loads(z.read('REUSE_AUDIT.json'))['cells']==20
        print('ADMISSION_SOURCE_HASHES_48_INITIALIZERS_20_STRICT_REUSES_PASS',sha(args.admission))
    if args.package:
        tree=ast.parse((HERE/'orchestrate.py').read_text());assigns={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['BASIC_ALLOW','ADMISSION_ALLOW']}
        expected=set(assigns['BASIC_ALLOW'])|{f'admission_seed{s}/{n}' for s in [42,43,44] for n in assigns['ADMISSION_ALLOW']}
        with zipfile.ZipFile(args.package) as z:
            assert set(z.namelist())==expected|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert manifest['schema']=='LOCO5_3SEED_V1' and manifest['private_artifacts_exported'] is False
            assert set(manifest['files'])==expected
            for n,h in manifest['files'].items():
                assert hashlib.sha256(z.read(n)).hexdigest()==h
                text=z.read(n).decode();assert not any(s in text for s in ['fit_patient_ids','password=','PRIVATE KEY'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(text.splitlines()));assert rows
                    assert not set(rows[0])&{'patient','patient_id','channel','channel_id','y','probability','prediction','path','archive_indices'}
            for n,h in manifest['source'].items():assert sha(HERE/n)==h
            q=json.loads(z.read('INDEPENDENT_VERIFICATION.json'))
            assert q['status']=='PASS' and q['patient_rows']==1200 and q['metric_values']==10800 and q['nominal_intervals']==405 and q['family_intervals']==8
            assert q['cells']==60 and q['reused']==20 and q['fresh']==40
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_3SEED'
            for n,count in {'MAIN_RESULTS.csv':5,'SOURCE_RESULTS.csv':20,'SEED_RESULTS.csv':15,'SOURCE_SEED_RESULTS.csv':60,'SEED_MEAN_SD.csv':5,'SOURCE_SEED_SD.csv':20,'POOLED_SEED_RESULTS.csv':75,'INTERVALS.csv':405,'PRIMARY_FAMILY_INTERVALS.csv':8,'TRAIN_CELLS.csv':60,'COVERAGE.csv':60,'VALIDATION_REPLAY.csv':60}.items():
                assert len(list(csv.DictReader(z.read(n).decode().splitlines())))==count,n
            rows=list(csv.DictReader(z.read('TRAIN_CELLS.csv').decode().splitlines()))
            assert sum(r['reused']=='True' for r in rows)==20
            assert {(int(r['train_seed']),r['arm'],int(r['held_index'])) for r in rows}=={(s,a,k) for s in [42,43,44] for a in ['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE','LINEAR_SVM_FEATURE_PR','TIMECONVCNN_RAW'] for k in range(1,5)}
        print('MULTISEED_ANONYMOUS_PACKAGE_SOURCE_HASH_COUNTS_PASS',sha(args.package))
    print('MULTISEED_CPU_RELEASE_PASS')
if __name__=='__main__':main()
