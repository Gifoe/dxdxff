"""Local aggregate/numeric checks plus all protected previous artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    root=HERE/'results/qualified';manifest=jr(root/'PUBLIC_TRANSFER_MANIFEST.json')
    assert len(manifest['files'])==6 and manifest['status']=='PASS'
    for name,entry in manifest['files'].items():
        assert (root/name).stat().st_size==entry['bytes'] and sha(root/name)==entry['sha256']
    assert sha(HERE/'export_public.py')==manifest['exporter_sha256']
    assert sha(HERE/'verify_qualification.py')==manifest['verifier_sha256']
    assert sha(HERE/'results/UNIT_GATE_FAILURE_DIAGNOSIS.json')==manifest['failed_original_gate_sha256']
    status=jr(root/'QUALIFICATION_STATUS.json');verified=jr(root/'INDEPENDENT_QUALIFICATION_VERIFICATION.json')
    assert verified['status']=='PASS' and verified['independently_reaggregated_coordinate_cells']==91620
    assert verified['inner_numeric_cells_checked']==450 and status['source_scalar_replay_max_abs_error']==0
    assert status['classifier_fits']==0 and not status['labels_indexed'] and not status['new_Outer_access']
    assert status['eligible_groups']==6 and not status['goal_stable_minimum_or_predictive_increment_proven']
    cv=pd.read_csv(root/'SOURCE_COVERAGE.csv');overall=cv[cv.center=='ALL']
    assert len(cv)==30 and len(overall)==6
    assert (overall.channels==7635).all() and (overall.patients==80).all() and (overall.record_channel_units==24995).all()
    ro=pd.read_csv(root/'INNER_NUMERIC_QUALIFICATION.csv');assert len(ro)==450
    assert ro.all_fitted_inputs_finite.all() and ro.imputer_and_scaler_selection_train_only.all()
    dec=pd.read_csv(root/'QUALIFICATION_DECISIONS.csv');assert len(dec)==6 and dec.eligible_for_future_preregistered_supervised_comparison.all()
    assert not dec.predictive_usefulness_proven.any()
    stress=pd.read_csv(root/'UNIT_STRESS.csv');assert (stress.unit_equivariant_max_abs_error<=1e-9).all()
    bad={'y','labels','labels_ez','probability','prediction','predictions'}
    for name in ['measurement.py','qualify.py','verify_qualification.py','constant_safe.py','diagnose_unit_gate.py']:
        for node in ast.walk(ast.parse((HERE/name).read_text(encoding='utf-8'))):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):assert node.slice.value not in bad
    old=HERE.parent/'m8_physiology_discovery_selection_seed42_v2';oldmanifest=jr(old/'RELEASE_MANIFEST.json')
    for name,entry in oldmanifest['files'].items():assert sha(old/name)==entry['sha256']
    protected=0
    for folder in ['m8_rawconditional_feature_selection_seed42_v1/results/dense_reference',
                   'm8_conditional_readout_validation_seed42_v1/results/readout',
                   'm8_nonlocal8_eligibility_audit_seed42_v1/results']:
        base=HERE.parent/folder
        for name,entry in jr(base/'PUBLIC_TRANSFER_MANIFEST.json')['files'].items():assert sha(base/name)==entry['sha256'];protected+=1
    result=dict(status='PASS',actual_public_aggregate_files=6,old_v2_files_unchanged=len(oldmanifest['files']),
        previous_supplement_public_files_unchanged=protected,all_six_measurements_eligible=True,
        original_unit_gate_failure_preserved=True,classifier_fits=0,new_Outer_access=False,goal_fully_achieved=False)
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
