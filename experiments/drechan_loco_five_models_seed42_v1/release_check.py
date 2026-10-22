"""CPU release/source/package verifier; no server identities or data required."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);args=parser.parse_args()
    for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
    q=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert q.returncode==0
    proofpath=HERE/'public_admission/ADMISSION.json';proof=json.loads(proofpath.read_text()) if proofpath.exists() else None
    if proof:
        assert proof['status']=='PASS' and proof['initialization_cells']==16 and proof['no_test_inference']
        for n,h in proof['source'].items():assert sha(HERE/n)==h,(n,'SOURCE_CHANGED_AFTER_ADMISSION')
    if args.package:
        assert proof is not None
        tree=ast.parse((HERE/'pipeline.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets));allow=ast.literal_eval(node.value)
        with zipfile.ZipFile(args.package) as z:
            assert len(z.namelist())==len(allow)+1 and set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert manifest['private_artifacts_exported'] is False
            assert manifest['source']==proof['source'] and set(manifest['files'])==set(allow)
            assert json.loads(z.read('ADMISSION.json'))==proof
            for n in allow:
                assert hashlib.sha256(z.read(n)).hexdigest()==manifest['files'][n] and z.getinfo(n).file_size<5000000
                text=z.read(n).decode('utf-8');assert not any(s in text for s in ['fit_patient_ids','password=','PRIVATE KEY'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(text.splitlines()));assert rows
                    assert not set(rows[0])&{'patient','patient_id','channel','channel_id','y','probability','prediction','path','archive_indices'}
            v=json.loads(z.read('INDEPENDENT_VERIFICATION.json'));assert v['status']=='PASS' and v['patient_rows']==400 and v['metric_values']==3600 and v['nominal_intervals']==405 and v['family_intervals']==8
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_SEED42'
            for n,count in {'MAIN_RESULTS.csv':5,'SOURCE_RESULTS.csv':20,'SOURCE_EQUAL_RESULTS.csv':5,'SOURCE_SD.csv':5,'POOLED_RESULTS.csv':25,'INTERVALS.csv':405,'PRIMARY_FAMILY_INTERVALS.csv':8,'TRAIN_CELLS.csv':20,'ROLE_AUDIT.csv':4,'COVERAGE.csv':20,'VALIDATION_REPLAY.csv':20,'INITIALIZATION_AUDIT.csv':16}.items():
                assert len(list(csv.DictReader(z.read(n).decode().splitlines())))==count,n
        print('LOCO_ANONYMOUS_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(args.package))
    print('LOCO_RELEASE_SOURCE_AND_CPU_TESTS_PASS')
if __name__=='__main__':main()
