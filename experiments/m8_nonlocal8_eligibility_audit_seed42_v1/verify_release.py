"""Verify anonymous audit bytes, decisions and untouched earlier releases."""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=HERE/'results';manifest=json.loads((data/'PUBLIC_TRANSFER_MANIFEST.json').read_text())
    assert manifest['status']=='PASS' and len(manifest['files'])==9
    for name,value in manifest['files'].items():
        assert (data/name).stat().st_size==value['bytes'] and sha(data/name)==value['sha256'],name
    assert sha(HERE/'export_public.py')==manifest['source_sha256']
    status=json.loads((data/'AUDIT_STATUS.json').read_text())
    assert status['model_fits']==0 and not status['labels_used'] and not status['new_Outer_access']
    for key,value in dict(patients=80,canonical_channels=7635,records=256,record_channel_units=24995,
                          valid_channel_windows=1471965,valid_record_windows=15074).items():assert status[key]==value
    verified=json.loads((data/'INDEPENDENT_AUDIT_VERIFICATION.json').read_text());assert verified['status']=='PASS'
    assert verified['private_feature_record_rows_reaggregated']==2048 and verified['permutation_rows_reaggregated']==4096
    assert verified['distribution_summary_cells_checked']==192
    coverage=pd.read_csv(data/'ALIGNMENT_COVERAGE.csv').set_index('center');assert coverage.loc['ALL','native_vs_Raw_mask_disagreement_record_windows']==27
    for key in coverage.columns:
        if key.startswith('min_'):continue
        assert coverage.loc['ALL',key]==coverage.drop('ALL')[key].sum()
    formula=pd.read_csv(data/'FORMULA_MASK_SUMMARY.csv').query("scope == 'overall'")
    assert len(formula)==8 and (formula.formula_exact_records==256).all() and (formula.formula_max_abs==0).all()
    decisions=pd.read_csv(data/'ELIGIBILITY_DECISIONS.csv').set_index('feature')
    assert decisions.eligible_for_later_distinct_candidate_admission.sum()==6
    assert not decisions.biological_ineffectiveness_claim.any() and not decisions.training_or_feature_pool_change_authorized.any()
    permutation=pd.read_csv(data/'PERMUTATION_EQUIVARIANCE.csv').query("scope == 'overall'")
    rank=[n for n in decisions.index if n.startswith('onset_rank_')]
    for row in permutation.itertuples(index=False):
        assert row.records==256 and row.units==24995
        assert (row.permutation_violating_units>0)==(row.feature in rank)
        if row.feature in rank:assert row.permutation_max_abs==1
    # Ensure this audit never indexes categorical clinical labels or outcomes.
    banned={'y','labels','labels_ez','patient_index','probability','prediction','predictions'}
    for name in ['audit.py','nonlocal_helpers.py','verify_audit.py']:
        tree=ast.parse((HERE/name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):assert node.slice.value not in banned
    previous=HERE.parent/'m8_physiology_discovery_selection_seed42_v2'
    old=json.loads((previous/'RELEASE_MANIFEST.json').read_text())
    for name,value in old['files'].items():assert sha(previous/name)==value['sha256']
    protected=0
    for folder in ['m8_rawconditional_feature_selection_seed42_v1/results/dense_reference',
                   'm8_conditional_readout_validation_seed42_v1/results/readout']:
        root=HERE.parent/folder;entry=json.loads((root/'PUBLIC_TRANSFER_MANIFEST.json').read_text())
        for name,value in entry['files'].items():assert sha(root/name)==value['sha256'];protected+=1
    result=dict(status='PASS',source_audit_public_files=9,old_v2_release_files_unchanged=len(old['files']),
        old_conditional_and_readout_public_files_unchanged=protected,
        actual_model_fits=0,new_Outer_inference=False,all8_cached_formulas_replay_exact=True,
        six_distinct_Raw250_scalar_operators_computable=True,two_rank_operators_rejected_as_order_dependent=True,
        stable_minimum_or_Raw_gain_proven=False,source_metadata_scope_not_clinical_provenance=True)
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
