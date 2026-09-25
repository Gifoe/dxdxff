from __future__ import annotations
from . import MODEL_VERSION,LABEL_DIRECTION_VERSION
def protocol(folds,seed): return {'pipeline':'TRACE-RawMIL','model_version':MODEL_VERSION,'label_direction_version':LABEL_DIRECTION_VERSION,'nez_label':1,'ez_label':0,'success_label':1,'failure_label':0,'center_as_input':False,'patient_id_as_input':False,'outer_test_used_for_training':False,'primary_threshold':.5,'outer_folds':int(folds.outer_fold.nunique()),'seed':seed,'paper_valid':False}
