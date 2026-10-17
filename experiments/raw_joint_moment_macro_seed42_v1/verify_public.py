"""Public source/artifact consistency; not a substitute for private replay."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    arms=['NATIVE','FROZEN_PRE','PRE_MEAN_DUP','PRE_ENERGY_DUP','PRE_JOINT'];assert protocol['arms']==arms
    result=HERE/'results/internal_v1';completed=(result/'INTERNAL_TERMINAL.json').exists()
    if completed:
        terminal=json.loads((result/'INTERNAL_TERMINAL.json').read_text())
        audit=json.loads((result/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert terminal['status']=='COMPLETE_VERIFIED_JOINT_RAW_INTERNAL' and audit['status']=='PASS'
        assert terminal['actual_neural_fits']==audit['actual_neural_fits'] and 15<=terminal['actual_neural_fits']<=45
        assert terminal['new_Outer_passes']==0 and not terminal['goal_established']
        assert audit['independent_CI_metric_checks']==126 and audit['maximum_bootstrap_difference']<1e-12
        cells=table(result/'INTERNAL_CELL_RESULTS.csv');means=table(result/'FIRST_PATH_MAIN.csv');folds=table(result/'FIRST_PATH_FOLD.csv')
        assert len(means)==5 and len(folds)==25 and {r['arm'] for r in means}==set(arms)
        assert len(table(result/'TRAIN_HISTORY.csv'))==terminal['epochs']
        trained=[r for r in cells if r['arm'] in arms[2:]]
        assert len(trained)==terminal['actual_neural_fits'] and all(int(r['selected_epoch'])>=1 and int(r['parameters'])==14577 for r in trained)
        points={r['arm']:r for r in means}
        for r in means:
            cc={q['fold']:q for q in cells if q['arm']==r['arm'] and q['screen']=='1'}
            ff=[q for q in folds if q['arm']==r['arm']];assert len(cc)==len(ff)==5
            total=sum(int(q['patients']) for q in cc.values());assert total==86
            for m in METRICS:
                value=math.fsum(float(q[m])*int(cc[q['fold']]['patients']) for q in ff)/total
                assert abs(value-float(r[m]))<1e-12
        for r in table(result/'FIRST_PATH_PAIRED_CI.csv'):
            assert int(r['appearances'])==86 and int(r['unique_patients'])==59 and int(r['draws'])==10000
            assert abs(float(r['mean'])-(float(points[r['left']][r['metric']])-float(points[r['right']][r['metric']])))<1e-12
        assert len(table(result/'OWN_FIT_SELECTION.csv'))==5
    final=HERE/'results/final_stage_v1';final_verified=(final/'INDEPENDENT_FINAL_VERIFICATION.json').exists()
    if final_verified:
        assert completed
        audit=json.loads((final/'INDEPENDENT_FINAL_VERIFICATION.json').read_text())
        terminal=json.loads((final/'FINAL_TERMINAL.json').read_text())
        support=json.loads((final/'RAW_SUPPORT_AUDIT.json').read_text())
        assert audit['status']==support['status']=='PASS' and audit['patient_metric_rows']==480
        assert audit['fresh_fits']==15 and audit['logical_cells']==30 and audit['independent_CI_metric_checks']==126
        assert audit['maximum_bootstrap_difference']<1e-12
        assert terminal['status']=='COMPLETE_SIX_JOINT_RAW_PROCEDURES' and terminal['primary']=='SELECTED'
        assert terminal['scientific_status']=='EXPLORATORY_REPEATED_TEST' and not terminal['independent_confirmation']
        rows=table(final/'OUTER_PATIENT_MEAN.csv');folds=table(final/'OUTER_FOLDS.csv')
        assert len(rows)==6 and len(folds)==30 and {r['procedure'] for r in rows}==set(arms+['SELECTED'])
        counts={str(k):n for k,n in zip(range(1,6),[16,16,17,15,16])}
        for r in rows:
            ff=[q for q in folds if q['procedure']==r['procedure']];assert len(ff)==5
            for m in METRICS:assert abs(math.fsum(float(q[m])*counts[q['fold']] for q in ff)/80-float(r[m]))<1e-12
            assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
        primary=next(float(r['macro_f1']) for r in rows if r['procedure']=='SELECTED')
        assert terminal['primary_point_over_065']==(primary>.65)
        coverage=table(final/'OUTER_METRIC_COVERAGE.csv');assert len(coverage)==54
        assert all(int(r['finite_patients'])==int(r['total_patients'])==80 for r in coverage)
        assert [support['counts'][k] for k in ['patients','canonical_channels','seizure_records','valid_windows']]==[80,7635,256,1471965]
        assert support['no_training_or_prediction'] and support['no_masks_or_data_changed']
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','raw','score_nez','logit_nez','pred_ez','y'}
    for path in (HERE/'results').rglob('*.csv'):
        with path.open(encoding='utf-8',newline='') as handle:assert not(set(csv.DictReader(handle).fieldnames)&forbidden),path.name
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes')
                 and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    manifest=dict(status='PASS',scope='public_consistency_not_private_Raw_replay',internal_verified=completed,final_verified=final_verified,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    destination=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:destination.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(destination.read_text(encoding='utf-8'))==manifest,'Public artifact hash mismatch'
    print('PUBLIC_JOINT_RAW_DELIVERY_PASS',len(paths),'files;internal_verified',completed,'final_verified',final_verified)


if __name__=='__main__':main()
