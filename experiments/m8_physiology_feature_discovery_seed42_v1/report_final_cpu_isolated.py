"""Execute exact frozen report AST in a CPU-only process, never import model/Torch."""
import ast
import faulthandler
faulthandler.enable()
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss,balanced_accuracy_score

TASK=Path(__file__).resolve().parent
OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
ROOT=OUT/'final_Outer';PUB=ROOT/'public'
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import sha,read,save,write,load
EV=load('isolated_frozen_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))


def main():
    assert 'torch' not in sys.modules
    fr=json.loads((ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json').read_text());fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    seal=json.loads((ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json').read_text());assert seal['freeze_sha256']==fh and len(seal['files'])==130
    source=TASK/'evaluate_final_v2.py';assert sha(source)==fr['binding']['source'][source.name]
    assert sha(EV.__file__)==fr['binding']['metric_engine_sha256'];cells={}
    for r in fr['selections']:
        name=r['model'];fold=r['fold'];p=ROOT/'armfold_scores_private'/f'{name}_fold{fold}.pkl'
        assert sha(p)==seal['files'][name+'_'+str(fold)];q=read(p)
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        cells[name,fold]=q
    rec=read(BASELINE/'RECONSTRUCTION_PRIVATE.pkl');assert 'torch' not in sys.modules
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['classification','report']]
    assert [n.name for n in nodes]==['classification','report']
    module=ast.Module(body=nodes,type_ignores=[]);code=compile(module,str(source),'exec');namespace=dict(globals())
    exec(code,namespace)
    print('CPU_ISOLATED_EXACT_FROZEN_REPORT_AST_ALL130_SEALS_NO_TORCH_NO_NEW_PREDICTION',flush=True)
    namespace['report'](cells,fr,rec,fh)
    assert 'torch' not in sys.modules
    write(PUB/'REPORTING_RECOVERY_AUDIT.json',dict(status='PASS',reporter_sha256=sha(__file__),
        exact_frozen_report_and_classification_AST=True,original_ranking_function_unmodified=True,torch_not_imported=True,
        all130_score_hashes_verified=True,no_model_creation_training_or_new_inference=True,thresholds_models_scores_and_metrics_unchanged=True,
        preceding_native_failures='NumPy_lexsort_and_pandas_grouping_in_model_importing_processes',
        lower_level_native_cause_not_proven=True,numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,
        frozen_evaluator_sha256=sha(source),metric_engine_sha256=sha(EV.__file__)))
    print('ALL26_FROZEN_OUTER_REPORTS_COMPLETE_CPU_ISOLATED_EXACT_CODE_NO_NEW_INFERENCE',flush=True)


if __name__=='__main__':main()
