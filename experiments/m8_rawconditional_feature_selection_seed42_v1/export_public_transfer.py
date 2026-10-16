"""Whitelisted anonymous public-result hashes, not a private runtime archive."""
from conditional_io import *

NAMES = [
    'ALL20_ACTUAL_RIDGE_ADMISSION.json', 'ASSESSMENT_CARDINALITY.csv',
    'ASSESSMENT_SOURCE.csv', 'CANONICAL_ADMISSION_CONTEXT.json',
    'CONDITIONAL_SELECTION_STABILITY.csv', 'ENCODER_STATUS.json',
    'FROZEN_INTERNAL_SELECTION.csv', 'INTERNAL_FOLD_RESULTS.csv',
    'INTERNAL_MAIN_RESULTS.csv', 'INTERNAL_PAIRED_INTERVALS.csv',
    'INTERNAL_REPLAY_VERIFICATION.json', 'INTERNAL_REPORT_STATUS.json',
    'INTERNAL_SOURCE_RESULTS.csv', 'NUMERICAL_REPAIR_EXECUTION.json',
    'SCREEN_STATUS.json', 'SEARCH_RECORD.csv', 'WEIGHT_BOOTSTRAP_RESELECTION.csv',
    'CHOICE_CONTROL_COMPARISON.csv', 'CHOICE_CONTROL_STATUS.json',
]


if __name__ == '__main__':
    pub = Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime\public')
    assert jr(pub / 'INTERNAL_REPLAY_VERIFICATION.json')['status'] == 'PASS'
    assert jr(pub / 'INTERNAL_REPORT_STATUS.json')['status'] == 'COMPLETE_INTERNAL_ONLY_CONDITIONAL_SCREEN_VERIFIED'
    assert jr(pub / 'CHOICE_CONTROL_STATUS.json')['status'] == 'COMPLETE_REGISTERED_CHOICE_CONTROLS'
    frozen = jr(Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime') / 'INPUT_FREEZE_PRIVATE.json')
    assert bindings() == frozen['original_scientific_binding']
    write(pub / 'PUBLIC_TRANSFER_MANIFEST.json', dict(status='PASS',
        original_v2_prediction_seal_unchanged=True, source_sha256=sha(Path(__file__)),
        files={n:dict(bytes=(pub/n).stat().st_size, sha256=sha(pub/n)) for n in NAMES}))
    print('ANONYMOUS_PUBLIC_TRANSFER_MANIFEST_COMPLETE', len(NAMES), flush=True)
