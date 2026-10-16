"""Whitelist-only anonymous source audit, never private freeze contents/identities."""
import platform
from common import *


if __name__=='__main__':
    end=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');train=jr(ROOT/'MODEL_FREEZE_PRIVATE.json')
    status=jr(PUB/'OUTER_STATUS.json');verify=jr(PUB/'INDEPENDENT_RESULTS_VERIFICATION.json')
    assert status['status']=='ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED' and verify['status']=='PASS'
    assert status['freeze_sha256']==sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for name,value in train['binding']['source'].items():assert sha(HERE/name)==value
    for name,value in end['evaluation_source'].items():assert sha(HERE/name)==value
    source=dict(stage1=train['binding']['stage1']['source'],training=train['binding']['source'],evaluation=end['evaluation_source'])
    write(PUB/'SCIENTIFIC_SOURCE_AUDIT.json',dict(status='PASS',source=source,
        data_reconstruction_sha256=EXPECTED_REC,raw_registry_sha256=EXPECTED_REG,
        old_inner_encoder_registry_sha256=EXPECTED_ENCODERS,
        own_FIT_recipe_freeze_sha256=sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json'),
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),
        pre_Outer_freeze_sha256=status['freeze_sha256'],score_seal_sha256=status['score_seal_sha256'],
        cohort=dict(patients=80,canonical_channels=7635,seizures=256,valid_raw_windows=1471965),
        Python=platform.python_version(),Torch=torch.__version__,NumPy=np.__version__,
        scikit_learn=train['binding']['scikit_learn'],precision='FP32_TF32_OFF',
        models=20,formal_epochs=224,new_Outer_passes=1,
        identifiers_paths_features_labels_scores_or_weights_exported=False))
    target=jr(PUB/'TARGET_STATUS.json')
    write(PUB/'RESEARCH_TERMINAL.json',dict(status='COMPUTATION_COMPLETE_TARGET_NOT_MET' if not target['point_target_met'] else 'COMPUTATION_COMPLETE_POINT_TARGET_MET_EXPLORATORY',
        target_metric=target['metric'],target=.65,observed=target['observed'],
        no_post_Outer_model_threshold_or_selector_changes=True,
        individual_component_end_to_end_effects_not_fully_identified=True,
        no_positive_attention_increment_established=True,
        objective_achievement_not_implied_by_computational_completion=True))
    print('WHITELIST_ONLY_SCIENTIFIC_AUDIT_EXPORTED',flush=True)
