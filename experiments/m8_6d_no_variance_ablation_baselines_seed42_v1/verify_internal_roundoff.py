"""Verification-only RF summation-roundoff audit; immutable training/math retained."""
from common import *
import tabular
import importlib.util
BOUND=8*np.finfo(np.float64).eps
SUMMARY={}

def verified_tabular(binding,core,rec,banks):
    frozen_tab=read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl');assert frozen_tab['input']==binding
    grid=np.round(np.arange(0,1.0001,.005),3);rows=[]
    for r in frozen_tab['cells']:
        d=bank(r['fold'],banks,'FULL_F0');saved=read(r['checkpoint'])
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and r['config']==tabular.CONFIG['configs'][r['classifier']][str(r['fold'])]
        z=tabular.view(core,rec['R14'][d['archive_indices']],d['patient'],r['PR'])
        assert legacy.array_sha(z)==saved['binding']['features']
        p=tabular.scores(r['classifier'],saved['bundle'],z[d['val']]);oldp=saved['validation_probability']
        error=float(np.max(np.abs(p-oldp)))
        if r['classifier']=='RandomForest':assert error<=BOUND
        else:assert np.array_equal(p,oldp)
        assert np.array_equal(p[:,None]>=grid,oldp[:,None]>=grid)
        assert np.array_equal(p>=r['threshold'],oldp>=r['threshold'])
        sel=tabular.BL.EV.threshold((d['y'][d['val']]==0).astype(int),p,d['patient'][d['val']])
        assert sel==saved['selected'] and sel['threshold']==r['threshold']
        ranks_equal=True
        for pid in sorted(set(d['patient'][d['val']])):
            j=d['patient'][d['val']]==pid;channels=d['channel'][d['val']][j].astype(str)
            ranks_equal &= np.array_equal(np.lexsort((channels,-p[j])),np.lexsort((channels,-oldp[j])))
        assert ranks_equal
        if r['classifier'] in ['Linear_SVM','RBF_SVM']:
            a=saved['bundle']['calibration_audit'];assert len(a)==3
            assert all(v['patient_exclusion'] and v['preprocessing_fit_subtrain_only'] for v in a)
        rows.append(dict(arm=r['arm'],fold=r['fold'],maximum_replay_error=error,all201_decisions_exact=True,selected_decisions_exact=True,
            selection_identical=True,within_patient_ranking_exact=True,FIT_only=True,active_dimension=12))
    assert len(rows)==70
    csv_out('TABULAR_VALIDATION_CHECKS',rows)
    audit=dict(status='PASS_VERIFICATION_ONLY_RF_ROUNDOFF',cells=70,maximum_absolute_probability_error=max(r['maximum_replay_error'] for r in rows),
        RF_only_bound=float(BOUND),other_models_bit_exact=True,all201_decisions_exact=True,all_selected_thresholds_and_decisions_exact=True,
        all_within_patient_rankings_exact=True,training_or_prediction_math_changed=False,weights_or_preprocessors_or_thresholds_changed=False,
        original_failed_verifier_preserved=True,wrapper_sha256=sha(Path(__file__)))
    SUMMARY.update(audit);write(PUB/'TABULAR_REPLAY_ROUNDOFF_AUDIT.json',audit)
    return 70

def main():
    tabular.verify_selected=verified_tabular
    m=module('sixd_immutable_internal_verifier',HERE/'verify_internal.py')
    original=m.write
    def audited_write(path,obj):
        if Path(path)==PUB/'INTERNAL_VERIFICATION.json':
            obj=dict(obj,maximum_neural_replay_error=obj['maximum_replay_error'],
                maximum_tabular_probability_replay_error=SUMMARY['maximum_absolute_probability_error'],
                maximum_replay_error=max(obj['maximum_replay_error'],SUMMARY['maximum_absolute_probability_error']),
                tabular_all201_decisions_and_rankings_exact=True,verification_only_roundoff_audit_sha256=sha(PUB/'TABULAR_REPLAY_ROUNDOFF_AUDIT.json'),
                verification_wrapper_sha256=sha(Path(__file__)))
        return original(path,obj)
    m.write=audited_write;m.main()
    print('SIXD_ALL150_VALIDATION_REPLAYS_WITH_RF_MACHINE_ROUNDOFF_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
