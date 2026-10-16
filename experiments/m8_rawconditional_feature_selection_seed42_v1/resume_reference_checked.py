"""Checked terminal-only recovery/verification of the frozen reference execution.

Never launches encoder training or repeats a completed canonical screen. The
original numerical wrapper and its binding remain byte-identical.
"""
import runpy
from conditional_io import *


def main():
    destination = Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')
    pub = destination / 'public'
    # FileShare.None coordination is supplied by run_dense_reference.ps1 for the
    # initial screen. This recovery entry refuses absent/partial fold products.
    b = jr(destination / 'INPUT_FREEZE_PRIVATE.json')
    assert b['original_scientific_binding'] == bindings()
    for name, value in b['numerical_engine'].items():
        assert sha(HERE / name) == value
    for fold in range(1, 6):
        assert read(destination / f'fold{fold}_SCREEN_PRIVATE.pkl')['binding'] == b
    assert len(jr(destination / 'SELECTED_FEATURES_PRIVATE.json')['folds']) == 5
    if not (pub / 'CANONICAL_ADMISSION_CONTEXT.json').exists():
        runpy.run_path(str(HERE / 'verify_dense_reference.py'), run_name='__main__')
    assert jr(pub / 'CANONICAL_ADMISSION_CONTEXT.json')['status'] == 'PASS'
    if jr(pub / 'SCREEN_STATUS.json').get('status') != 'COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN':
        runpy.run_path(str(HERE / 'finalize_dense_metadata.py'), run_name='__main__')
    if not (pub / 'INTERNAL_REPORT_STATUS.json').exists():
        runpy.run_path(str(HERE / 'verify_and_report_internal.py'), run_name='__main__')
    assert jr(pub / 'INTERNAL_REPLAY_VERIFICATION.json')['status'] == 'PASS'
    assert jr(pub / 'INTERNAL_REPORT_STATUS.json')['status'] == 'COMPLETE_INTERNAL_ONLY_CONDITIONAL_SCREEN_VERIFIED'
    if not (pub / 'CHOICE_CONTROL_STATUS.json').exists():
        runpy.run_path(str(HERE / 'export_choice_controls.py'), run_name='__main__')
    assert jr(pub / 'CHOICE_CONTROL_STATUS.json')['status'] == 'COMPLETE_REGISTERED_CHOICE_CONTROLS'
    print('CANONICAL_INTERNAL_ARTIFACTS_COMPLETE_NO_OUTER', flush=True)


if __name__ == '__main__':
    main()
