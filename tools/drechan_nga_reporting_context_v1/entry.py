"""Reporting-only context admission; never training, prediction or source mutation."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import psutil

HERE=Path(__file__).resolve().parent
CODE=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1')
ROOT=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1_runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['admit','report_outer','verify_outer','package_public']
    stage=sys.argv[1];assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:assert os.environ.get(n)=='1'
    psutil.Process().cpu_affinity(list(range(16,32)))
    manifest=json.loads((HERE/'EXECUTION_MANIFEST.json').read_text())
    for n,h in manifest['execution_source'].items():assert sha(HERE/n)==h
    for n,h in manifest['scientific_source'].items():assert sha(CODE/n)==h
    sys.path.insert(0,str(CODE));import common as c
    fr=c.frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    seal=c.jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json');sh=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert fh==manifest['freeze_sha256'] and sh==manifest['score_seal_sha256'] and len(seal['files'])==255
    assert seal['freeze']==fh and seal['all255_cells_sealed_before_metric_reporting']
    for n,h in seal['files'].items():assert sha(ROOT/'scores_private'/(n+'.pkl'))==h
    before_public={p.name:sha(p) for p in c.PUB.iterdir() if p.is_file()}
    patient=ROOT/'PATIENT_METRICS_PRIVATE.pkl'
    before_private=sha(patient) if patient.exists() else None
    # The original report load_rows and verifier call frozen(), not setup().
    # Supplying the ORIGINAL setup initializes roots for historical/reuse audit.
    binding,*_=c.setup()
    assert binding==fr['input'] and sys.modules['common'] is c
    assert set(c.CONTEXT['roots'])=={'parent','complete3','poolcap'}
    if stage!='admit':
        assert c.jr(c.PUB/'REPORTING_CONTEXT_REPAIR_ADMISSION.json')['status']=='PASS'
        sys.argv=[str(CODE/(stage+'.py'))]
        runpy.run_path(sys.argv[0],run_name='__main__')
    for n,h in manifest['scientific_source'].items():assert sha(CODE/n)==h
    assert sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==fh and sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==sh
    for n,h in seal['files'].items():assert sha(ROOT/'scores_private'/(n+'.pkl'))==h
    prefix_exact=True
    if stage in ['admit','report_outer']:
        for n,h in before_public.items():assert sha(c.PUB/n)==h,('EXISTING_AGGREGATE_CHANGED',n)
        if before_private is not None:assert sha(patient)==before_private
    if stage=='report_outer':assert c.jr(c.PUB/'REPORT_STATUS.json')['status']=='ALL17_FIXED_GROUPS_SELECTED3SEEDS_REPORTED'
    if stage in ['verify_outer','package_public']:
        assert c.jr(c.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
        assert c.jr(c.PUB/'OUTER_STATUS.json')['status']=='COMPLETE_VERIFIED_NGA_17_GROUP_SELECTED3SEED_FIVE_FOLD_OUTER'
    label={'admit':'ADMISSION','report_outer':'REPORT','verify_outer':'VERIFY','package_public':'PACKAGE'}[stage]
    c.write(c.PUB/('REPORTING_CONTEXT_REPAIR_'+label+'.json'),dict(status='PASS',stage=stage,
        execution_manifest_sha256=sha(HERE/'EXECUTION_MANIFEST.json'),scientific14_sources_unchanged=True,
        freeze_sha256=fh,score_seal_sha256=sh,sealed_score_files_checked=255,sealed_scores_bit_exact=True,
        existing_aggregate_prefix_bit_exact=prefix_exact,existing_patient_metrics_bit_exact=True,
        existing_aggregate_prefix_files=len(before_public),original_input_binding_exact=True,
        original_context_setup_only=True,no_training_or_predictions=True,no_metric_math_or_threshold_changes=True))
    print('NGA_REPORTING_CONTEXT_REPAIR_PASS',stage,flush=True)
if __name__=='__main__':main()
