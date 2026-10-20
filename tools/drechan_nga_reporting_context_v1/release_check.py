"""Release-only verification of reporting-context proof and immutable source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
EXP=HERE.parent.parent/'experiments/drechan_no_guidance_ablation_42_44_62_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);args=parser.parse_args()
    manifest=json.loads((HERE/'EXECUTION_MANIFEST.json').read_text())
    admission=json.loads((EXP/'public_admission/ADMISSION.json').read_text())
    assert manifest['scientific_source']==admission['source'] and len(manifest['scientific_source'])==14
    for n,h in manifest['scientific_source'].items():assert sha(EXP/n)==h
    for n,h in manifest['execution_source'].items():assert sha(HERE/n)==h
    stages={'ADMISSION':'admit','REPORT':'report_outer','VERIFY':'verify_outer','PACKAGE':'package_public'}
    for label,stage in stages.items():
        path=HERE/'audit'/('REPORTING_CONTEXT_REPAIR_'+label+'.json')
        proof=json.loads(path.read_text())
        assert proof['status']=='PASS' and proof['stage']==stage
        assert proof['execution_manifest_sha256']==sha(HERE/'EXECUTION_MANIFEST.json')
        assert proof['freeze_sha256']==manifest['freeze_sha256'] and proof['score_seal_sha256']==manifest['score_seal_sha256']
        assert proof['scientific14_sources_unchanged'] and proof['sealed_score_files_checked']==255 and proof['sealed_scores_bit_exact']
        assert proof['original_input_binding_exact'] and proof['original_context_setup_only']
        assert proof['no_training_or_predictions'] and proof['no_metric_math_or_threshold_changes']
        if label in ['ADMISSION','REPORT']:
            assert proof['existing_aggregate_prefix_bit_exact'] and proof['existing_patient_metrics_bit_exact']
    cmd=[sys.executable,str(EXP/'release_check.py')]
    if args.package:cmd+=['--package',str(args.package)]
    assert subprocess.run(cmd).returncode==0
    print('REPORTING_CONTEXT_ONLY_SOURCE_SCORE_PREFIX_AND_RELEASE_PROOFS_PASS')
if __name__=='__main__':main()
