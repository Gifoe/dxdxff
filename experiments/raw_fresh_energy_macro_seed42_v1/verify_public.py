"""Public consistency/hash audit; private direct model/Raw replays are separate."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;INNER=HERE/'results/internal_v1';FINAL=HERE/'results/final_stage_v1'
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(path):
    with path.open(encoding='utf-8',newline='') as h:return list(csv.DictReader(h))


def main():
    terminal=json.loads((INNER/'INTERNAL_TERMINAL.json').read_text());audit=json.loads((INNER/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
    assert terminal['status']=='COMPLETE_VERIFIED_FRESH_ENERGY_INTERNAL' and audit['status']=='PASS'
    assert terminal['actual_neural_fits']==42 and terminal['epochs']==1062 and audit['patient_metric_rows']==1157
    assert terminal['new_Outer_passes']==0 and terminal['advanced_folds']==[1,3,4,5] and audit['maximum_replay_error']==0
    cells=table(INNER/'INTERNAL_CELL_RESULTS.csv');means=table(INNER/'FIRST_PATH_MAIN.csv');folds=table(INNER/'FIRST_PATH_FOLD.csv')
    assert len(cells)==68 and len(means)==6 and len(folds)==30 and len(table(INNER/'TRAIN_HISTORY.csv'))==1062
    trained=[r for r in cells if r['arm'] not in ['NATIVE','FROZEN_PRE']]
    assert len(trained)==42 and all(int(r['selected_epoch'])>=1 for r in trained) and sum(int(r['epochs']) for r in trained)==1062
    points={r['arm']:r for r in means}
    for r in means:
        cc=sorted([q for q in cells if q['arm']==r['arm'] and q['screen']=='1'],key=lambda q:int(q['fold']))
        ff=sorted([q for q in folds if q['arm']==r['arm']],key=lambda q:int(q['fold']))
        assert sum(int(q['patients']) for q in cc)==86
        for m in METRICS:assert abs(np.average([float(q[m]) for q in ff],weights=[int(q['patients']) for q in cc])-float(r[m]))<1e-12
    for r in table(INNER/'FIRST_PATH_PAIRED_CI.csv'):
        assert int(r['appearances'])==86 and int(r['unique_patients'])==59 and int(r['draws'])==10000
        assert abs(float(r['mean'])-(float(points[r['left']][r['metric']])-float(points[r['right']][r['metric']])))<1e-12
    for r in table(INNER/'FIRST_PATH_FACTORIAL_INTERACTION_CI.csv'):
        m=r['metric'];expected=float(points['PRE_LOGENERGY'][m])-float(points['PRE_MEAN'][m])-float(points['WINDOW_LOGENERGY'][m])+float(points['WINDOW_MEAN'][m])
        assert abs(expected-float(r['mean']))<1e-12
    final_verified=(FINAL/'INDEPENDENT_FINAL_VERIFICATION.json').exists()
    if final_verified:
        v=json.loads((FINAL/'INDEPENDENT_FINAL_VERIFICATION.json').read_text());assert v['status']=='PASS' and v['patient_metric_rows']==560
        rows=table(FINAL/'OUTER_PATIENT_MEAN.csv');ff=table(FINAL/'OUTER_FOLDS.csv');assert len(rows)==7 and len(ff)==35
        for r in rows:
            g=sorted([q for q in ff if q['procedure']==r['procedure']],key=lambda q:int(q['fold']))
            for m in METRICS:assert abs(np.average([float(q[m]) for q in g],weights=[16,16,17,15,16])-float(r[m]))<1e-12
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','raw','score_nez','logit_nez','pred_ez','y'}
    for p in (HERE/'results').rglob('*.csv'):
        with p.open(encoding='utf-8',newline='') as h:assert not (set(csv.DictReader(h).fieldnames)&forbidden),p.name
    paths=sorted([p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes') and '__pycache__' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json'])
    manifest=dict(status='PASS',scope='public_consistency_not_private_Raw_replay',internal_verified=True,final_verified=final_verified,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    path=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(path.read_text(encoding='utf-8'))==manifest,'Artifact hash mismatch'
    print('PUBLIC_FRESH_RAW_DELIVERY_PASS',len(paths),'files;final_verified',final_verified)


if __name__=='__main__':main()
