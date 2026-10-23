"""Independent anonymous release aggregation and coverage checks, no private data."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
REPO=Path(__file__).resolve().parents[2]
EXP=REPO/'experiments/drechan_loco_five_models_3seed_v1';PUB=EXP/'public'
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
SEEDS=[42,43,44]
ARMS=['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE','LINEAR_SVM_FEATURE_PR','TIMECONVCNN_RAW']
SOURCES=['hup','lzu','multicenter','pediatric'];N=dict(zip(SOURCES,[36,21,15,8]))
def read(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<1e-12,(a,b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((PUB/'PACKAGE_MANIFEST.json').read_bytes())
assert len(manifest['files'])==34 and manifest['private_artifacts_exported'] is False
for n,h in manifest['files'].items():assert sha(PUB/n)==h,n
for n,h in manifest['source'].items():assert sha(EXP/n)==h,n
assert len(manifest['source'])==12
cells=read('TRAIN_CELLS');assert len(cells)==60
assert {(int(r['train_seed']),r['arm'],int(r['held_index'])) for r in cells}=={(s,a,k) for s in SEEDS for a in ARMS for k in range(1,5)}
assert sum(r['reused']=='True' for r in cells)==20
assert all((r['reused']=='True')==(r['train_seed']=='42') for r in cells)
assert all(6<=int(r['epochs'])<=30 for r in cells if r['arm']!='LINEAR_SVM_FEATURE_PR')
assert all(r['epochs']=='0' for r in cells if r['arm']=='LINEAR_SVM_FEATURE_PR')
coverage=read('COVERAGE');assert len(coverage)==60
for s in SEEDS:
    for arm in ARMS:
        rows=[r for r in coverage if int(r['train_seed'])==s and r['arm']==arm]
        assert len(rows)==4 and {r['held_source'] for r in rows}==set(SOURCES)
        assert sum(int(r['patients']) for r in rows)==80 and sum(int(r['channels']) for r in rows)==7635
        assert all(int(r['patients'])==N[r['held_source']] for r in rows)
        raw=arm in ['OURS_RAW_FEATURE','OURS_RAW','TIMECONVCNN_RAW']
        assert all((r['raw_consumed']=='True')==raw for r in rows)
        assert sum(int(r['valid_raw_windows']) for r in rows)==(1471965 if raw else 0)
seed=read('SEED_RESULTS');srcseed=read('SOURCE_SEED_RESULTS');main=read('MAIN_RESULTS');summary=read('SEED_MEAN_SD')
src=read('SOURCE_RESULTS');equal=read('SOURCE_EQUAL_RESULTS');sd=read('SOURCE_SD')
assert len(seed)==15 and len(srcseed)==60 and len(main)==len(summary)==len(equal)==len(sd)==5
for arm in ARMS:
    per=[r for r in seed if r['arm']==arm];assert {int(r['train_seed']) for r in per}==set(SEEDS)
    a=next(r for r in main if r['arm']==arm);b=next(r for r in summary if r['arm']==arm)
    c=next(r for r in equal if r['arm']==arm);d=next(r for r in sd if r['arm']==arm)
    sources=[r for r in src if r['arm']==arm];assert len(sources)==4
    for m in METRICS:
        values=[float(r[m]) for r in per];close(a[m],statistics.mean(values));close(b[m+'_mean'],a[m])
        close(b[m+'_sample_sd'],statistics.stdev(values))
        v=[float(r[m]) for r in sources];close(c[m],statistics.mean(v));close(d[m],statistics.stdev(v))
        close(a[m],sum(float(r[m])*N[r['held_source']] for r in sources)/80)
    for r in sources:
        rows=[q for q in srcseed if q['arm']==arm and q['held_source']==r['held_source']]
        assert {int(q['train_seed']) for q in rows}==set(SEEDS)
        for m in METRICS:close(r[m],statistics.mean(float(q[m]) for q in rows))
family=read('PRIMARY_FAMILY_INTERVALS');assert len(family)==8
for r in family:
    control=r['comparison'][len('FULL_MINUS_'):];m=r['metric']
    a=next(q for q in main if q['arm']==ARMS[0]);b=next(q for q in main if q['arm']==control)
    close(r['estimate'],float(a[m])-float(b[m]))
    assert r['family_tests']=='8' and r['draws']=='10000' and r['seed']=='42'
lock=json.loads((EXP/'PARENT_LOCK.json').read_text());endings=0
for remote,h in {**lock['dependencies'],**lock['extra_dependencies']}.items():
    tail=remote.replace('\\','/')[3:]
    if remote.startswith('E:'):tail=tail.split('DRE-nips/new-pipeline/7-11/',1)[1]
    path=next((p for p in [REPO/'experiments'/tail,REPO/'dre/experiments'/tail] if p.is_file()),None)
    assert path is not None,tail
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=h:
        lf=data.replace(b'\r\n',b'\n');assert h in [hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b'\n',b'\r\n')).hexdigest()]
        endings+=1
assert len(lock['dependencies'])+len(lock['extra_dependencies'])==58
supp=EXP/'public_supplement/REPORTING_IO_REPAIR_PUBLIC.json'
audit=json.loads(supp.read_bytes());assert audit['status']=='PASS' and audit['frozen_scientific_files']==12
assert all(audit[k] for k in ['all_scientific_source_hashes_unchanged','all60_checkpoint_validation_score_bindings_unchanged','original_computed_metric_tables_byte_identical','no_training_or_inference_rerun','no_threshold_or_model_change'])
helper=(REPO/'tools/drechan_loco_3seed_reporting_repair_v1/reporting_entry.py').read_bytes()
lf=helper.replace(b'\r\n',b'\n')
assert audit['helper_sha256'] in [hashlib.sha256(helper).hexdigest(),hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b'\n',b'\r\n')).hexdigest()]
print('ANONYMOUS_AGGREGATE_COVERAGE_SEED_MEAN_SAMPLE_SD_SOURCE_WEIGHTING_60CELL_34PAYLOAD_58PARENT_SOURCE_PASS',dict(line_ending_only=endings))
