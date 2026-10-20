"""Release-only source/explicit-public-package/test checks, no scientific changes."""
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
    proofpath=HERE/'public_admission/ADMISSION.json';proof=None
    if proofpath.exists():
        proof=json.loads(proofpath.read_text());assert proof['status']=='PASS'
        assert proof['expected_fresh_fits']==205 and proof['expected_reused_cells']==50 and proof['initialization_cells_checked']==255
        assert proof['all50_reuse_checks_bit_exact'] and proof['P2_complete_support_label_free_backward_checks']==320 and proof['seed_set_outcome_informed']
        for n,h in proof['source'].items():assert sha(HERE/n)==h,(n,'SOURCE_CHANGED_AFTER_ADMISSION')
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert r.returncode==0
    for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
    if args.package:
        assert proof
        tree=ast.parse((HERE/'package_public.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets))
        allow=ast.literal_eval(node.value);assert len(allow)==36
        with zipfile.ZipFile(args.package) as z:
            assert len(z.namelist())==37 and set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));assert not manifest['private_artifacts_exported'] and set(manifest['files'])==set(allow)
            for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h and z.getinfo(n).file_size<5000000
            verify=json.loads(z.read('INDEPENDENT_OUTER_VERIFICATION.json'))
            assert verify['status']=='PASS' and verify['patient_rows']==4080 and verify['patient_metric_checks']==36720
            assert verify['nominal_interval_checks']==306 and verify['primary_family_interval_checks']==32
            assert json.loads(z.read('OUTER_STATUS.json'))['status']=='COMPLETE_VERIFIED_NGA_17_GROUP_SELECTED3SEED_FIVE_FOLD_OUTER'
            assert json.loads(z.read('ADMISSION.json'))==proof
            counts={'MAIN_RESULTS.csv':17,'SEED_RESULTS.csv':51,'SEED_SD.csv':17,'FOLD_RESULTS.csv':255,'FOLD_SD.csv':51,
                'METRIC_COVERAGE.csv':459,'TRAIN_CELLS.csv':255,'RAW_COVERAGE.csv':255,'INTERVALS.csv':306,
                'PRIMARY_FAMILY_INTERVALS.csv':32,'COMPONENT_EVIDENCE.csv':16,'FOLD_SEED_DIRECTIONS.csv':240,
                'ROLE_AUDIT.csv':5,'INITIALIZATION_AUDIT.csv':255,'ADMISSION_REUSE.csv':50,'POOLING_SUPPORT_ADMISSION.csv':5,
                'VALIDATION_REPLAY_CHECKS.csv':255,'PREPROCESSING_REPLAY.csv':5,'SEED_SELECTION_AUDIT.csv':5,
                'POOLED_RESULTS.csv':51,'POOLED_SEED_MEAN.csv':17}
            forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
            for n in allow:
                text=z.read(n).decode('utf-8');assert not any(s in text for s in ['password=','PRIVATE KEY','fit_patient_ids'])
                if n.endswith('.csv'):
                    rows=list(csv.DictReader(text.splitlines()));assert rows and not set(rows[0])&forbidden
                    if n in counts:assert len(rows)==counts[n],n
            completed=list(csv.DictReader(z.read('TRAIN_CELLS.csv').decode('utf-8').splitlines()))
            assert sum(r['reused']=='True' for r in completed)==50 and sum(r['reused']=='False' for r in completed)==205
            assert {(r['arm'],int(r['seed']),int(r['fold'])) for r in completed}=={(a,s,f) for a in json.loads((HERE/'PROTOCOL_LOCK.json').read_text())['arms'] for s in [42,44,62] for f in range(1,6)}
        print('EXPLICIT_ANONYMOUS_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(args.package))
    print('EXECUTABLE_TESTS_AND_AVAILABLE_SOURCE_LOCK_PASS')
if __name__=='__main__':main()
