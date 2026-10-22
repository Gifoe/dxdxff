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
        if HERE.parent.name=='experiments':
            repo=HERE.parents[1];checked=endings=0
            for remote,expected in proof['binding']['dependencies'].items():
                tail=remote.replace('\\','/')[3:]
                if remote.startswith('E:'):tail=tail.split('DRE-nips/new-pipeline/7-11/',1)[1]
                candidates=[repo/'experiments'/tail,repo/'dre/experiments'/tail]
                local=next((p for p in candidates if p.is_file()),None);assert local is not None,tail
                data=local.read_bytes()
                if hashlib.sha256(data).hexdigest()!=expected:
                    lf=data.replace(b'\r\n',b'\n')
                    assert expected in [hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b'\n',b'\r\n')).hexdigest()],(tail,'DEPENDENCY_CODE_CHANGED')
                    endings+=1
                checked+=1
            assert checked==56
            print('ALL56_IMMUTABLE_DEPENDENCIES_PRESENT_CODE_IDENTICAL',dict(line_ending_only_differences=endings))
            supplement=HERE/'public_supplement/PARENT_SOURCE_AUDIT.json'
            if supplement.exists():
                extra=json.loads(supplement.read_text());assert extra['status']=='PASS' and extra['phase']=='postrun_release_supplement'
                assert not extra['scientific_models_or_scores_changed'] and not extra['original_anonymous_package_changed']
                historical=json.loads((repo/extra['historical_admission']).read_text())
                assert historical['status']=='PASS' and len(extra['files'])==2
                for rel,h in extra['files'].items():assert sha(repo/rel)==h==historical['source'][Path(rel).name]
                print('TWO_ADDITIONAL_HISTORICALLY_FROZEN_PARENT_SOURCES_PASS_NOT_A_NEW_PREFIT_TIMESTAMP')
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
            lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());arms=lock['arms'];centers=lock['sources']
            cells=list(csv.DictReader(z.read('TRAIN_CELLS.csv').decode().splitlines()))
            assert {(r['arm'],int(r['held_index'])) for r in cells}=={(a,k) for a in arms for k in range(1,5)}
            assert all(r['reused']=='False' and r['held_source']==centers[int(r['held_index'])-1] for r in cells)
            assert sum(int(r['epochs']) for r in cells)==201
            coverage=list(csv.DictReader(z.read('COVERAGE.csv').decode().splitlines()))
            for arm in arms:
                rows=[r for r in coverage if r['arm']==arm]
                assert {r['held_source'] for r in rows}==set(centers)
                assert sum(int(r['patients']) for r in rows)==80 and sum(int(r['channels']) for r in rows)==7635
                raw=arm in ['OURS_RAW_FEATURE','OURS_RAW','TIMECONVCNN_RAW']
                assert all((r['raw_consumed']=='True')==raw for r in rows)
                assert sum(int(r['valid_raw_windows']) for r in rows)==(1471965 if raw else 0)
            freeze=json.loads(z.read('FREEZE_STATUS.json'));sealed=json.loads(z.read('PREDICTION_STATUS.json'));outer=json.loads(z.read('OUTER_STATUS.json'))
            assert freeze['freeze_sha256']==outer['freeze_sha256'] and sealed['seal_sha256']==outer['seal_sha256']
        print('LOCO_ANONYMOUS_PACKAGE_HASH_SCHEMA_COUNTS_PASS',sha(args.package))
    print('LOCO_RELEASE_SOURCE_AND_CPU_TESTS_PASS')
if __name__=='__main__':main()
