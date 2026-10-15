"""CPU reporting recovery from immutable130-cell seal; no model creation/inference."""
import faulthandler
faulthandler.enable()
from evaluate_final_v2 import ROOT,PUB,report
from integrated_data_v2 import *

def main():
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for name,h in fr['binding']['source'].items():assert sha(TASK/name)==h
    seal=jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert seal['freeze_sha256']==fh and len(seal['files'])==130
    if (PUB/'STATUS.json').exists() and jr(PUB/'STATUS.json')['status']=='ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED':
        print('SEALED_REPORT_ALREADY_COMPLETE_NO_RECOMPUTATION',flush=True);return
    print('REPORT_ONLY_VERIFY_ALL130_SEALED_FILES',flush=True);cells={}
    for r in fr['selections']:
        name=r['model'];fold=r['fold'];p=ROOT/'armfold_scores_private'/f'{name}_fold{fold}.pkl'
        assert sha(p)==seal['files'][name+'_'+str(fold)]
        q=read(p);assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        assert 'y' not in q and np.isfinite(q['score_nez']).all();cells[name,fold]=q
    rec=read(BASELINE/'RECONSTRUCTION_PRIVATE.pkl');assert len(cells)==130
    print('REPORT_ONLY_ALL_SCORES_VALID_CALL_UNCHANGED_FROZEN_REPORT',flush=True)
    report(cells,fr,rec,fh)
    write(PUB/'REPORTING_RECOVERY_AUDIT.json',dict(status='PASS',reporter_sha256=sha(__file__),frozen_evaluator_sha256=sha(TASK/'evaluate_final_v2.py'),
        seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),all130_score_hashes_verified=True,new_model_creation_or_inference=False,
        thresholds_models_scores_and_metric_function_unchanged=True,preceding_process_native_exit='0xC0000005',preceding_fault_module='unknown',
        preceding_evaluator_completed_all130_scores_before_exit=True))
    print('ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED_FROM_SEAL_NO_NEW_INFERENCE',flush=True)


if __name__=='__main__':main()
