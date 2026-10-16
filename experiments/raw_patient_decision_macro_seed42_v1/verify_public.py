"""Check anonymous delivery schemas, weighted fold means and immutable file hashes."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(path):
    with path.open(encoding='utf-8',newline='') as h:return list(csv.DictReader(h))


def main():
    internal=HERE/'results/internal_v1';final=HERE/'results/final_stage_v1'
    assert json.loads((internal/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())['status']=='PASS'
    verification=json.loads((final/'INDEPENDENT_FINAL_VERIFICATION.json').read_text())
    assert verification['status']=='PASS' and verification['patient_rows']==720 and verification['valid_raw250_windows']==1471965
    assert json.loads((final/'REPORTING_VERIFIER_REPAIR.json').read_text())['status']=='PASS'
    means=table(final/'OUTER_PATIENT_MEAN.csv');folds=table(final/'OUTER_FOLDS.csv');sizes=np.array([16,16,17,15,16])
    assert len(means)==9 and len(folds)==45
    for row in means:
        part=sorted([r for r in folds if r['procedure']==row['procedure']],key=lambda r:int(r['fold']))
        for m in METRICS:assert abs(np.average([float(r[m]) for r in part],weights=sizes)-float(row[m]))<1e-12
    original=next(r for r in means if r['procedure']=='NATIVE__ORIGINAL')
    assert abs(float(original['macro_f1'])-.6304930529317453)<1e-12
    assert not any(float(r['macro_f1'])>.65 for r in means)
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','pred_ez','logit_nez','score_nez','y'}
    for path in (HERE/'results').rglob('*.csv'):
        assert not (set(table(path)[0])&forbidden),path.name
    paths=sorted([p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes') and '__pycache__' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json'])
    manifest=dict(status='PASS',patient_equal_Macro_primary=True,verified_procedures=9,goal_point_established=False,
                  files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    target=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(target.read_text(encoding='utf-8'))==manifest,'Published artifact hash mismatch'
    print('PUBLIC_DELIVERY_PASS',len(paths),'files;goal_not_achieved')


if __name__=='__main__':main()
