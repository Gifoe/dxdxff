from __future__ import annotations
from . import MODEL_VERSION,PROTOCOL
def ensure_unique(rows,expected=None):
 keys=[x['patient_key'] for x in rows]
 if len(keys)!=len(set(keys)):raise RuntimeError('duplicate OOF patient_key')
 if expected is not None and len(keys)!=expected:raise RuntimeError(f'OOF count {len(keys)} != cache count {expected}')
def protocol(folds):return {'model_version':MODEL_VERSION,'protocol':PROTOCOL,'uses_task1':False,'uses_task1_checkpoint':False,'uses_task1_embedding':False,'uses_ez_nez_labels':False,'uses_ez_nez_routing':False,'uses_channel_auxiliary_loss':False,'uses_self_supervised_pretraining':True,'uses_single_dev_validation':True,'uses_early_stopping':True,'checkpoint_selection_metric':'validation_ema_bce','outer_test_inference_count_per_fold':1,'folds':folds}
