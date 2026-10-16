"""Resume only terminal metadata after completed screen's nested-binding KeyError."""
from conditional_io import *

if __name__=='__main__':
    dest=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime');pub=dest/'public'
    b=jr(dest/'INPUT_FREEZE_PRIVATE.json');assert jr(dest/'SCREEN_INPUT_FREEZE_PRIVATE.json')==b
    original=b['original_scientific_binding'];assert original==bindings()
    for name,h in b['numerical_engine'].items():assert sha(HERE/name)==h
    assert jr(pub/'ALL20_ACTUAL_RIDGE_ADMISSION.json')['status']=='PASS'
    assert jr(pub/'CANONICAL_ADMISSION_CONTEXT.json')['status']=='PASS'
    f=jr(dest/'SELECTED_FEATURES_PRIVATE.json');assert f['binding']==b and len(f['folds'])==5
    for fold in range(1,6):assert read(dest/f'fold{fold}_SCREEN_PRIVATE.pkl')['binding']==b
    metrics=read(dest/'ASSESSMENT_PATIENT_METRICS_PRIVATE.pkl');assert len(metrics)==2805
    search=pd.read_csv(pub/'SEARCH_RECORD.csv');assert search.fold.nunique()==5
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==original['parent_Outer_seal_protection']
    write(pub/'SCREEN_STATUS.json',dict(status='COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN',actual_ridge_fits=len(search)+165,
        assessment_rows=len(metrics),per_fold_k={fold:v['k'] for fold,v in f['folds'].items()},
        original_validation_and_Outer_used=False,new_neural_candidate_or_Outer_inference=False,positive_minimality_or_increment_not_assumed=True))
    write(pub/'NUMERICAL_REPAIR_EXECUTION.json',dict(status='SCREEN_RECOMPUTED_PENDING_INDEPENDENT_VERIFICATION',
        no_Raw_encoder_retraining=True,original_failed_screen_and_all_checkpoints_preserved=True,numerical_engine=b['numerical_engine'],
        terminal_nested_binding_KeyError_repaired_in_metadata_only=True,no_head_fit_or_prediction_repeated_by_this_finalizer=True,
        original_v2_Outer_seal_unchanged=True,source_sha256=sha(Path(__file__))))
    print('ALL_COMPLETED_SCREEN_METADATA_RESUMED_WITHOUT_REPEATING_HEADS',len(search)+165,flush=True)
