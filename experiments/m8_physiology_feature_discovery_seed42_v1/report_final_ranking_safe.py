"""Recover aggregation only, with equivalent sorting; all frozen scientific files untouched."""
import faulthandler
faulthandler.enable()
import evaluate_final_v2 as R
from integrated_data_v2 import *
from ranking_safe import ranking


def main():
    unit=jr(OUT/'public/RANKING_SORT_EQUIVALENCE.json');assert unit['status']=='PASS'
    assert unit['source_sha256']==sha(TASK/'ranking_safe.py') and unit['test_sha256']==sha(TASK/'test_ranking_safe.py')
    fr=jr(R.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');fh=sha(R.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for n,h in fr['binding']['source'].items():assert sha(TASK/n)==h
    seal=jr(R.ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert seal['freeze_sha256']==fh and len(seal['files'])==130
    cells={}
    for r in fr['selections']:
        name=r['model'];fold=r['fold'];p=R.ROOT/'armfold_scores_private'/f'{name}_fold{fold}.pkl'
        assert sha(p)==seal['files'][name+'_'+str(fold)];q=read(p)
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        assert 'y' not in q;cells[name,fold]=q
    original=R.EV.ranking
    try:
        R.EV.ranking=ranking
        print('REPORT_ONLY_EXACT_CANONICAL_SORT_ALL130_SEALS_VALID_NO_NEW_PREDICTION',flush=True)
        R.report(cells,fr,read(BASELINE/'RECONSTRUCTION_PRIVATE.pkl'),fh)
    finally:R.EV.ranking=original
    write(R.PUB/'REPORTING_RECOVERY_AUDIT.json',dict(status='PASS',reporter_sha256=sha(__file__),ranking_sha256=sha(TASK/'ranking_safe.py'),
        ranking_unit_sha256=sha(OUT/'public/RANKING_SORT_EQUIVALENCE.json'),all130_score_hashes_verified=True,
        no_training_model_creation_or_new_prediction=True,thresholds_models_scores_unchanged=True,classification_and_bootstrap_functions_unchanged=True,
        ranking_rule_unchanged='descending_unrounded_EZ_score_then_ascending_canonical_channel_name',
        preceding_native_failure='NumPy_lexsort_access_violation_confirmed_by_faulthandler',frozen_sources_unchanged=True))
    print('ALL26_FROZEN_OUTER_REPORTS_COMPLETE_WITH_EQUIVALENT_RANKING_NO_NEW_INFERENCE',flush=True)


if __name__=='__main__':main()
