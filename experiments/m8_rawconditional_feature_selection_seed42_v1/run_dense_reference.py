"""Versioned numerical repair; reuse encoders, never overwrite original evidence."""
import conditional_io as io
from canonical_ridge import CanonicalRidge


def main():
    original=io.bindings();assert io.jr(io.ROOT/'INPUT_FREEZE_PRIVATE.json')==original
    assert io.jr(io.PUB/'ENCODER_STATUS.json')['status']=='ALL20_INNER_CROSSFIT_ENCODERS_COMPLETE'
    assert io.jr(io.PUB/'NUMERICAL_ADMISSION_STATUS.json')['status']=='FAIL'
    registry=io.read(io.ROOT/'ENCODER_REGISTRY_PRIVATE.pkl');assert registry['binding']==original
    destination=io.Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')
    destination.mkdir(exist_ok=True);(destination/'public').mkdir(exist_ok=True)
    binding=dict(original_scientific_binding=original,original_encoder_registry_sha256=io.sha(io.ROOT/'ENCODER_REGISTRY_PRIVATE.pkl'),
        numerical_engine={n:io.sha(io.HERE/n) for n in ['NUMERICAL_ENGINE_AMENDMENT.json','canonical_ridge.py','run_dense_reference.py']},
        scientific_preprocessing_roles_hyperparameters_unchanged=True,old_failed_outputs_retained=True)
    io.freeze(destination/'INPUT_FREEZE_PRIVATE.json',binding)
    if not (destination/'ENCODER_REGISTRY_PRIVATE.pkl').exists():io.save(destination/'ENCODER_REGISTRY_PRIVATE.pkl',dict(binding=binding,entries=registry['entries']))
    io.write(destination/'public/ENCODER_STATUS.json',dict(status='ALL20_INNER_CROSSFIT_ENCODERS_COMPLETE',complete_cells=20,
        reused_without_retraining=True,original_registry_sha256=binding['original_encoder_registry_sha256']))
    module=io.load('conditional_v1_canonical_reference_screen',io.HERE/'select_conditional.py')
    module.ROOT=destination;module.PUB=destination/'public';module.ConditionalRidge=CanonicalRidge;module.bindings=lambda:binding
    module.main()
    assert io.sha(io.OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==original['parent_Outer_seal_protection']
    io.write(destination/'public/NUMERICAL_REPAIR_EXECUTION.json',dict(status='SCREEN_RECOMPUTED_PENDING_INDEPENDENT_VERIFICATION',
        no_Raw_encoder_retraining=True,original_failed_screen_and_all_checkpoints_preserved=True,
        numerical_engine=binding['numerical_engine'],original_v2_Outer_seal_unchanged=True))


if __name__=='__main__':main()
