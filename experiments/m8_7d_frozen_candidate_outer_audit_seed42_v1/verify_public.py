"""Public-only schema and aggregate cross-check; no access to private data."""
import csv
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
def table(root,name):
    with (root/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    p=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());arms=p['arms'];metrics=p['metrics'];root=HERE/'results'
    s=json.loads((root/'OUTER_STATUS.json').read_text());v=json.loads((root/'INDEPENDENT_OUTER_VERIFICATION.json').read_text())
    assert s['status']=='COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT' and v['status']=='PASS'
    assert s['patient_rows']==v['patient_rows']==880 and s['metric_checks']==v['patient_metric_checks']==7920
    assert v['interval_checks']==234 and v['macro_family_interval_checks']==10 and v['maximum_interval_difference']<1e-12
    assert s['selected_arm'] is None and not s['independent_confirmation'] and s['new_inference_cells']==50 and s['reused_reference_cells']==5
    main=table(root,'MAIN_RESULTS');assert len(main)==11 and {r['arm'] for r in main}==set(arms);points={r['arm']:r for r in main}
    for row in main:
        assert int(row['patients'])==80
        assert abs(float(row['macro_f1'])-.5*(float(row['f1'])+float(row['nez_f1'])))<1e-12
        assert all(math.isfinite(float(row[m])) for m in metrics)
    assert abs(float(points['C2_REFERENCE']['macro_f1'])-.6396579900303192)<1e-12
    ff=table(root,'FOLD_RESULTS');assert len(ff)==55
    for arm in arms:
        z=[r for r in ff if r['arm']==arm];assert {int(r['fold']) for r in z}==set(range(1,6))
        assert sum(int(r['patients']) for r in z)==80
        for m in metrics:assert abs(math.fsum(float(r[m])*int(r['patients']) for r in z)/80-float(points[arm][m]))<1e-12
    ci=table(root,'INTERVALS');assert len(ci)==234
    for r in ci:
        mean=float(points[r['left']][r['metric']])-(float(points[r['right']][r['metric']]) if r['right'] else 0)
        assert abs(float(r['mean'])-mean)<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000
    assert len(table(root,'MACRO_FAMILY_INTERVALS'))==10
    raw=table(root,'RAW_COVERAGE');assert len(raw)==50
    for arm in arms[1:]:
        z=[r for r in raw if r['arm']==arm]
        assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635 and sum(int(r['valid_windows']) for r in z)==1471965
    frozen=table(root,'FROZEN_MODELS');assert len(frozen)==55
    assert {(r['arm'],int(r['fold'])) for r in frozen}=={(a,f) for a in arms for f in range(1,6)}
    assert all(len(r['checkpoint_sha256'])==64 and int(r['epoch'])>=1 and 0<=float(r['threshold'])<=1 for r in frozen)
    cov=table(root,'METRIC_COVERAGE');assert len(cov)==99 and all(int(r['finite_patients'])==int(r['total_patients'])==80 for r in cov)
    observed=json.loads((root/'OBSERVED_TARGETS.json').read_text());assert observed['selected_arm'] is None and not observed['independent_confirmation']
    for r in observed['arms']:
        point=float(points[r['arm']]['macro_f1']);assert abs(r['patient_equal_Macro_F1']-point)<1e-12 and r['above_0_64']==bool(point>.64)
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for path in root.glob('*.csv'):
        with path.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden),path.name
    print('ALL11_FROZEN_OUTER_PUBLIC_AGGREGATES_PASS')

if __name__=='__main__':main()
