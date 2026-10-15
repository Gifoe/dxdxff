"""Run unchanged independent metric/bootstrap suffix in parity-gated CPU runtime."""
import ast
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss,balanced_accuracy_score,matthews_corrcoef
TASK=Path(__file__).resolve().parent;OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime');ROOT=OUT/'final_Outer';PUB=ROOT/'public'
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import sha,read,write,load
EV=load('independent_cpu_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))
def jr(p):return json.loads(Path(p).read_text())

def main():
    assert 'torch' not in sys.modules
    parity=jr(OUT/'public/CPU_STATISTICS_RUNTIME_PARITY.json');assert parity['status']=='PASS'
    assert parity['source_sha256']==sha(TASK/'test_cpu_statistics_runtime_v2.py')
    audit=jr(PUB/'INDEPENDENT_FINAL_CACHE_VERIFICATION.json');assert audit['status']=='PASS'
    source=TASK/'verify_final.py';assert sha(source)==audit['original_verifier_sha256']
    context=ROOT/'VERIFIED_CACHE_CONTEXT_PRIVATE.pkl';assert sha(context)==audit['context_sha256'];q=read(context)
    assert q['original_verifier_sha256']==sha(source)
    tree=ast.parse(source.read_text());main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['equal','independent_classification']]
    assert len(helpers)==2
    cut=next(i for i,n in enumerate(main_node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='archive' for t in n.targets))
    namespace=dict(globals());namespace.update(q)
    exec(compile(ast.Module(body=helpers+main_node.body[cut:],type_ignores=[]),str(source),'exec'),namespace)
    assert 'torch' not in sys.modules
    result=jr(PUB/'INDEPENDENT_FINAL_VERIFICATION.json');assert result['status']=='PASS'
    result.update(cpu_metric_verifier_sha256=sha(__file__),original_verifier_sha256=sha(source),cache_audit_sha256=sha(PUB/'INDEPENDENT_FINAL_CACHE_VERIFICATION.json'),
        runtime_parity_sha256=sha(OUT/'public/CPU_STATISTICS_RUNTIME_PARITY.json'),metric_process_did_not_import_torch=True,
        metric_python=sys.version,metric_numpy=np.__version__,metric_pandas=pd.__version__,metric_sklearn=sklearn.__version__)
    write(PUB/'INDEPENDENT_FINAL_VERIFICATION.json',result)
    print('INDEPENDENT_CACHE_PLUS_CPU_ALL2080_METRICS_AND_BOOTSTRAPS_PASS',flush=True)


if __name__=='__main__':main()
