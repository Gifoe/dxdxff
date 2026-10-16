"""Anonymous delivery consistency only; private selected-state replay is separate."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RESULTS=HERE/'results'
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(name):
    with (RESULTS/name).open(encoding='utf-8',newline='') as h:return list(csv.DictReader(h))


def main():
    terminal=json.loads((RESULTS/'INTERNAL_TERMINAL.json').read_text());verified=json.loads((RESULTS/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
    assert terminal['status']=='COMPLETE_VERIFIED_TEMPORAL_CONTEXT_INTERNAL' and verified['status']=='PASS'
    assert terminal['actual_neural_fits']==15 and terminal['epochs']==126 and terminal['advanced_folds']==[]
    assert terminal['new_Outer_passes']==0 and not terminal['goal_established'] and not terminal['final_stage_warranted']
    assert verified['patient_metric_rows']==430 and verified['maximum_replay_error']==0
    cells=table('INTERNAL_CELL_RESULTS.csv');main=table('FIRST_PATH_MAIN.csv');folds=table('FIRST_PATH_FOLD.csv')
    trained=[r for r in cells if r['arm'] not in ['NATIVE','FROZEN_PRE']]
    assert len(cells)==25 and len(trained)==15 and sum(int(r['epochs']) for r in trained)==126
    assert sum(int(r['selected_epoch'])==0 for r in trained)==12
    assert len(table('TRAIN_HISTORY.csv'))==126 and len(main)==5 and len(folds)==25
    points={r['arm']:r for r in main}
    for row in main:
        cc=sorted([r for r in cells if r['arm']==row['arm']],key=lambda r:int(r['fold']))
        ff=sorted([r for r in folds if r['arm']==row['arm']],key=lambda r:int(r['fold']))
        assert sum(int(r['patients']) for r in cc)==86
        for m in METRICS:assert abs(np.average([float(r[m]) for r in ff],weights=[int(r['patients']) for r in cc])-float(row[m]))<1e-12
    for row in table('FIRST_PATH_PAIRED_CI.csv'):
        assert int(row['appearances'])==86 and int(row['unique_patients'])==59 and int(row['draws'])==10000
        expected=float(points[row['left']][row['metric']])-float(points[row['right']][row['metric']])
        assert abs(expected-float(row['mean']))<1e-12
    assert all(r['selected_arm']=='NATIVE' and r['advanced']=='False' for r in table('OWN_FIT_SELECTION.csv'))
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','pred_ez','logit_nez','score_nez','y','raw'}
    for p in RESULTS.glob('*.csv'):
        with p.open(encoding='utf-8',newline='') as h:assert not (set(csv.DictReader(h).fieldnames)&forbidden),p.name
    paths=sorted([p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes') and '__pycache__' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json'])
    manifest=dict(status='PASS',scope='anonymous_delivery_not_private_Raw_replay',new_Outer_passes=0,goal_established=False,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    path=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(path.read_text(encoding='utf-8'))==manifest,'Artifact hash mismatch'
    print('PUBLIC_TEMPORAL_DELIVERY_PASS',len(paths),'files;0newOuter;target_unmet')


if __name__=='__main__':main()
