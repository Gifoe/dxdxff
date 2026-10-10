"""Standard-library release gates over code hashes and non-identifying aggregates."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def js(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def rows(rel):
    with (ROOT/rel).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def near(a,b): assert math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=0),(a,b)

def main():
    required_audits='D2_REPRODUCTION.json GRAPH_SOURCE_AUDIT.json GRAPH_ALIGNMENT_AUDIT.json GRAPH_FEATURE_SCHEMA.json GRAPH_FEATURE_QUALITY.csv GRAPH_MISSINGNESS_AUDIT.csv GRAPH_REDUNDANCY_AUDIT.csv GRAPH_SOURCE_MODE.json GRAPH_SHUFFLE_AUDIT.json GRAPH_PREPROCESSING_AUDIT.json PARAMETER_AUDIT.json ENGINEERING_TESTS.json'.split()
    required_results='TRAINING_HISTORY.csv MODEL_SELECTION.csv VALIDATION_SUMMARY.csv VALIDATION_BY_FOLD.csv VALIDATION_BY_CENTER.csv PAIRED_BOOTSTRAP.csv GRAPH_DEPENDENCE_AUDIT.csv RANKING_CHANGE_AUDIT.csv ERROR_CORRECTION_AUDIT.csv EFFICIENCY_AUDIT.csv VALIDATION_GATE.json RUN_STATUS.json'.split()
    for name in required_audits: assert (ROOT/'audit'/name).is_file()
    for name in required_results: assert (ROOT/'results'/name).is_file()
    for name in ['README.md','SOURCE_AUDIT.md','PROTOCOL_LOCK.json','IMPLEMENTATION_AUDIT.md','FINAL_REPORT.md']: assert (ROOT/name).is_file()
    protocol=sha(ROOT/'PROTOCOL_LOCK.json'); audit=js('audit/ENGINEERING_TESTS.json')
    assert audit['status']=='PASS' and len(audit['checks'])==38 and len(audit['FIT_smokes'])==15
    assert all(x['status']=='PASS' for x in audit['checks']) and audit['protocol_sha256']==protocol
    assert not audit['formal_checkpoints_modified']
    for name,h in audit['code_sha256'].items(): assert sha(ROOT/'code'/name)==h,name
    run=js('results/RUN_STATUS.json'); gate=js('results/VALIDATION_GATE.json')
    assert run['status']=='COMPLETE' and run['registered_runs']==15 and run['all_runs_hash_verified']
    assert not run['outer_test_accessed'] and not gate['outer_test_accessed'] and gate['protocol_sha256']==protocol
    assert run['terminal']==gate['terminal']
    selections=rows('results/MODEL_SELECTION.csv'); assert len(selections)==15
    assert {(int(r['fold']),r['method']) for r in selections}=={(f,a) for f in range(1,6) for a in ['G1','G2','G3']}
    for f in range(1,6): assert len({r['initial_hash'] for r in selections if int(r['fold'])==f})==1
    for r in selections: assert 1<=int(r['selected_epoch'])<=int(r['epochs'])<=30
    summary={r['method']:r for r in rows('results/VALIDATION_SUMMARY.csv')}
    metrics=['macro_f1','ez_f1','nez_f1','balanced_accuracy','ez_auprc','ez_auroc','ez_mrr','top1_is_ez','sensitivity','specificity','accuracy']
    assert set(summary)=={'D2','G1','G2','G3','G1_GRAPH_ZERO'}
    for r in summary.values():
        assert int(r['patient_fold_appearances'])==65 and int(r['unique_patients'])==47
        for m in metrics: assert 0<=float(r[m])<=1
    repro=js('audit/D2_REPRODUCTION.json'); assert repro['status']=='PASS' and repro['verified_historical_files']==101
    for m in metrics: near(summary['D2'][m],repro['metrics'][m])
    byfold=rows('results/VALIDATION_BY_FOLD.csv'); assert len(byfold)==25
    for a in summary:
        for m in metrics: near(summary[a][m],sum(float(r[m]) for r in byfold if r['method']==a)/5)
    boots=rows('results/PAIRED_BOOTSTRAP.csv'); assert len(boots)==55
    for r in boots:
        left,right=r['contrast'].split('-'); near(r['delta'],float(summary[left][r['metric']])-float(summary[right][r['metric']]))
        assert int(r['draws'])==10000 and int(r['valid_draws'])==10000 and int(r['unique_patient_ids'])==47
        assert float(r['CI_low'])<=float(r['CI_high']) and 0<=int(r['positive_folds'])<=5
    history=rows('results/TRAINING_HISTORY.csv'); assert len(history)==sum(int(r['epochs']) for r in selections)
    schema=js('audit/GRAPH_FEATURE_SCHEMA.json'); assert schema['dimensions']==16 and not schema['degenerate']
    assert len(schema['names'])==16 and not schema['existing_graph_name_overlap']
    params=js('audit/PARAMETER_AUDIT.json'); assert params['parameter_counts']=={'D2':9221,'G1':10789,'G2':10789,'G3':10789}
    alignment=js('audit/GRAPH_ALIGNMENT_AUDIT.json'); assert alignment['status']=='PASS' and alignment['canonical_channels']==7635
    assert all(r['VAL_labels_used']=='False' for r in rows('audit/GRAPH_REDUNDANCY_AUDIT.csv'))
    # Headers may report population counts, not individual identities or scores.
    prohibited={'patient','patient_id','subject_id','channel','channel_name','canonical_channel_name','y','label','score','logit','prediction'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in files:
        assert p.suffix.lower() in {'.py','.md','.csv','.json',''} and '_PRIVATE' not in p.name,p
        assert p.stat().st_size<100000,p
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f: assert not prohibited.intersection(next(csv.reader(f))),p
        if p.suffix=='.json':
            def scan(x):
                if isinstance(x,dict):
                    assert not prohibited.intersection(x),p
                    for v in x.values(): scan(v)
                elif isinstance(x,list):
                    for v in x: scan(v)
            scan(json.loads(p.read_text(encoding='utf-8')))
    for number in range(1,24): assert '\n'+str(number)+'. ' in (ROOT/'FINAL_REPORT.md').read_text(encoding='utf-8')
    print('PUBLIC_DELIVERY_PASS',len(files),'files;',len(history),'epochs; exactly15 runs; 55 paired contrasts; no private row columns')

if __name__=='__main__': main()
