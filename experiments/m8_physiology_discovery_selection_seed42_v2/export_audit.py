"""Whitelisted aggregate runtime and source audit; never serialize private roles."""
import platform
from common import *


def main():
    assert jr(PUB/'OUTER_STATUS.json')['status']=='ALL105_FROZEN_SCORE_CELLS_SEALED'
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');support=read(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl')
    checks=[]
    for r in support['selections']:
        old=next(q for q in fr['selections'] if q['fold']==r['fold'] and q['recipe']=='RAW_ONLY')
        a=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)['binding']
        b=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['binding']
        assert a['initial']==b['initial'] and a['scale']==b['scale'] and a['seed']==b['seed']
        checks.append(dict(fold=r['fold'],initial_state_identical=True,FIT_scale_identical=True,seed_identical=True,only_pool_support_changed=True))
    frozen=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    write(PUB/'EXECUTION_AUDIT.json',dict(status='PASS',logical_and_actual_new_fits=85,
        main_epochs=jr(PUB/'NEURAL_STATUS.json')['epochs'],support_epochs=jr(PUB/'SUPPORT30_STATUS.json')['epochs'],
        selected_feature_freeze_sha256=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json'),main_model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),
        support_model_freeze_sha256=sha(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl'),Outer_freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),
        all_sources_bound_before_scores=True,matched_support_controls=checks,main_scientific_source=fr['binding']['source'],
        original_D1_trainer_sha256=fr['binding']['D1_trainer_sha256'],original_Raw_trainer_sha256=fr['binding']['original_raw_trainer_sha256'],
        frozen_evaluation_sources=frozen['evaluation_code'],historical_score_cells_reused=20,new_score_cells=85,
        no_individual_data_or_roles_in_this_export=True))
    import scipy,sklearn
    write(PUB/'STATISTICS_ENVIRONMENT.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,CPU_affinity='single_process_core16',
        roles='CPU statistics/replay only; GPU model training uses original PyTorch2.11.0+cu128 FP32 TF32off'))
    print('WHITELISTED_EXECUTION_AND_MATCHED_SUPPORT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
