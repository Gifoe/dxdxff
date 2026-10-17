"""Public aggregate consistency only,not a substitute for private frozen replay."""
import json
import math
from pathlib import Path
from verify_public import HERE,ARMS,METRICS,table


def main():
    root=HERE/'results/final_stage_v1'
    assert (root/'OUTER_STATUS.json').exists(),'No final results: unexecuted is not complete'
    status=json.loads((root/'OUTER_STATUS.json').read_text());v=json.loads((root/'INDEPENDENT_OUTER_VERIFICATION.json').read_text())
    decision=json.loads((root/'FINAL_DECISION.json').read_text())
    candidate=json.loads((HERE/'results/internal_v1/CANDIDATE_SELECTION.json').read_text())
    assert status['status']=='COMPLETE_VERIFIED_FULL80_PATIENT_MACRO_TRAINING' and v['status']=='PASS'
    selected=candidate['selected'];assert candidate['new_qualified'] and selected in ARMS[1:] and not candidate['new_Outer_used']
    assert status['selected']==decision['selected']==selected and status['features']==7 and status['patients']==80
    assert not status['independent_confirmation'] and status['scientific_status']=='EXPLORATORY_REPEATED_TEST'
    assert v['patient_rows']==240 and v['patient_metric_checks']==2160 and v['interval_checks']==54 and v['maximum_interval_difference']<1e-12
    assert v['original7_features_unchanged'] and v['one_classification_head'] and v['all_models_thresholds_frozen'] and v['old_scores_unchanged']
    rows=table(root/'MAIN_RESULTS.csv');points={z['arm']:z for z in rows};assert set(points)==set(ARMS)
    for z in rows:
        assert abs(float(z['macro_f1'])-.5*(float(z['f1'])+float(z['nez_f1'])))<1e-12
    a=float(points[selected]['macro_f1']);b=float(points[ARMS[0]]['macro_f1'])
    assert abs(b-.6396579900303192)<1e-12
    assert abs(a-status['selected_patient_Macro_F1'])<1e-12 and abs(b-status['control_patient_Macro_F1'])<1e-12
    target=a>.64 and a>b
    assert status['target_met']==target and (decision['status']=='POINT_TARGET_MET')==target
    assert abs(decision['patient_equal_Macro_F1']-a)<1e-12 and abs(decision['delta']-(a-b))<1e-12
    ff=table(root/'FOLD_RESULTS.csv');counts={'1':16,'2':16,'3':17,'4':15,'5':16}
    assert len(ff)==15
    for arm in ARMS:
        q=[z for z in ff if z['arm']==arm];assert {z['fold'] for z in q}==set(counts)
        for metric in METRICS:
            assert abs(math.fsum(float(z[metric])*counts[z['fold']] for z in q)/80-float(points[arm][metric]))<1e-12
    ci=table(root/'INTERVALS.csv');assert len(ci)==54
    for z in ci:
        expected=float(points[z['left']][z['metric']])-(float(points[z['right']][z['metric']]) if z['kind']=='paired' else 0)
        assert abs(float(z['mean'])-expected)<1e-12
        assert int(z['appearances'])==int(z['unique_patients'])==80 and int(z['draws'])==10000
    cov=table(root/'METRIC_COVERAGE.csv');assert len(cov)==27
    assert all(int(z['finite_patients'])==int(z['total_patients'])==80 for z in cov)
    raw=table(root/'RAW_COVERAGE.csv');assert len(raw)==10
    for arm in ARMS[1:]:
        q=[z for z in raw if z['arm']==arm];assert len(q)==5
        assert sum(int(z['patients']) for z in q)==80 and sum(int(z['channels']) for z in q)==7635
        assert sum(int(z['valid_windows']) for z in q)==1471965
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for path in root.glob('*.csv'):
        assert not(set(table(path)[0])&forbidden),path.name
    print('PATIENT_MACRO_TRAINING_FULL80_PUBLIC_AGGREGATES_PASS','target_met',target)


if __name__=='__main__':main()
