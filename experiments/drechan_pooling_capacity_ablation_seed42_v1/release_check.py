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
    repair=HERE/'public_admission/ENGINEERING_REPAIR_ADMISSION.json'
    if repair.exists():
        rp=json.loads(repair.read_text());assert rp['status']=='PASS' and rp['original13_sources_unchanged']
        assert rp['complete_patient_support_checks']==320 and rp['complete_patient_FIT_checks']==255 and rp['complete_patient_validation_checks']==65
        assert rp['zero_optimizer_updates'] and rp['no_Outer_access'] and rp['completed_epochs_discarded']==0
        execution=HERE.parent.parent/'tools/drechan_poolcap_masked_time_repair_v1'
        manifest=execution/'EXECUTION_MANIFEST.json';assert sha(manifest)==rp['execution_manifest_sha256']
        em=json.loads(manifest.read_text());assert em['scientific_source']==rp['scientific_source']==proof['source']
        for n,h in em['execution_source'].items():assert sha(execution/n)==h
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert result.returncode==0
    if a.package:
        assert admission.exists() and repair.exists()
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
            completed=list(csv.DictReader(z.read('TRAIN_CELLS.csv').decode('utf-8').splitlines()))
            retained=next(r for r in completed if r['arm']=='P1_TEMPORAL_STATS' and int(r['fold'])==1)
            prefix='models_private/P1_TEMPORAL_STATS/seed42/fold1/'
            assert retained['checkpoint_sha256']==rp['preserved_completed_P1_fold1_files'][prefix+'BEST_PRIVATE.pt']
            assert retained['validation_sha256']==rp['preserved_completed_P1_fold1_files'][prefix+'VALIDATION_PRIVATE.pkl']
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
