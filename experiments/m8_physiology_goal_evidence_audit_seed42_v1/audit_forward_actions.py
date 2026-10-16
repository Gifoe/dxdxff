"""Evaluate next-action constraints from actual published experiment states."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths={
        'main_Outer_status':ROOT/'m8_physiology_discovery_selection_seed42_v2/results/OUTER_STATUS.json',
        'main_support_status':ROOT/'m8_physiology_discovery_selection_seed42_v2/results/SUPPORT30_STATUS.json',
        'nonlinear_status':ROOT/'m8_conditional_readout_validation_seed42_v1/results/readout/READOUT_STATUS.json',
        'nonlinear_protocol':ROOT/'m8_conditional_readout_validation_seed42_v1/PROTOCOL_LOCK.json',
        'nonlinear_gate':ROOT/'m8_conditional_readout_validation_seed42_v1/results/readout/ALLOCATION_GATE.json',
        'six_group_status':ROOT/'m8_nonlocal6_conditional_selection_seed42_v1/results/REPORT_STATUS.json',
        'six_group_protocol':ROOT/'m8_nonlocal6_conditional_selection_seed42_v1/PROTOCOL_LOCK.json',
        'six_group_gate':ROOT/'m8_nonlocal6_conditional_selection_seed42_v1/results/ALLOCATION_GATE.json',
        'atlas_status':ROOT/'m8_physiological_evidence_atlas_seed42_v1/results/ATLAS_STATUS.json',
        'atlas_release':ROOT/'m8_physiological_evidence_atlas_seed42_v1/RELEASE_STATUS.json',
        'evidence_ledger':ROOT/'m8_physiological_evidence_atlas_seed42_v1/results/FEATURE_EVIDENCE_LEDGER_STATUS.json'}
    data={name:read(path) for name,path in paths.items()}
    assert data['main_Outer_status']['status']=='ALL105_FROZEN_SCORE_CELLS_SEALED'
    assert data['main_Outer_status']['scientific_status']=='EXPLORATORY_REPEATED_TEST'
    assert data['nonlinear_status']['status']=='ALL90_INTERNAL_READOUTS_SEALED_BEFORE_ASSESSMENT_REPORT'
    assert data['six_group_status']['status']=='COMPLETE_SEALED_INTERNAL_SIX_GROUP_SCREEN'
    assert data['atlas_status']['status']=='COMPLETE_DESCRIPTIVE_PHYSIOLOGICAL_EVIDENCE_ATLAS'
    assert data['atlas_release']['status']=='PASS'
    assert data['nonlinear_gate']['status']==data['six_group_gate']['status']=='FAIL'
    assert 'do not automatically add head variants' in data['nonlinear_protocol']['stop']
    assert 'Failure stops expansion' in data['six_group_protocol']['allocation_gate']
    assert data['nonlinear_gate']['no_full_Raw_M8_training_automatically_authorized']
    assert not data['six_group_gate']['automatic_full_M8_or_Outer_authorized']
    assert not data['evidence_ledger']['joint38_or76_coordinate_model_trained']
    assert not data['atlas_status']['stable_minimum_or_Raw_increment_proven']
    result=dict(status='EXTERNAL_RESEARCH_INPUT_OR_DECISION_REQUIRED',
        full_scientific_goal_achieved=False,goal_is_not_paused_or_marked_complete=True,
        current_goals_blocked_status_not_yet_set=True,first_consecutive_goal_turn_with_this_external_dependency=True,
        completed_Outer_score_cells=105,completed_nonlinear_readouts=90,
        completed_six_group_search_fits=data['six_group_status']['actual_search_fits'],
        completed_six_group_assessment_refits=data['six_group_status']['actual_assessment_refits'],
        both_registered_predictive_continuation_gates='FAIL',
        automatic_new_heads_seeds_hyperparameters_or_Outer=False,
        association_support_does_not_override_prediction_stopping_rules=True,
        joint38_global_minimum_claim=False,
        next_allowed_direction='A substantive new scientific input/decision and prospective protocol, not an automatic rescue of failed screens.',
        evidence_required_for_independent_confirmation='An unused patient cohort with verified labels, channel/time alignment and measurement provenance, not another read of already viewed Outer.',
        alternative_research_direction='A genuinely new, explicitly chosen hypothesis must be separately scoped and prospectively controlled; no guaranteed gain.',
        scope_boundary='No claim that other data do not exist on the server or in the world; no newly authorized independent cohort is present in these frozen experiment bindings.',
        new_model_fits=0,new_Outer_access=False,
        evidence={name:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for name,path in paths.items()})
    (HERE/'FORWARD_ACTION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':main()
