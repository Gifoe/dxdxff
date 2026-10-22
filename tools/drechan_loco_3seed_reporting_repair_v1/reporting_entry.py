"""Reporting I/O bridge only. Never changes frozen sources or reruns inference.

Deploy outside seed-named scientific roots, e.g. D:\\loco_reporting_release_v1.
The original csv_out function resolves common.PUB, not pipeline.PUB. Correct
that destination temporarily and preserve every frozen scientific function.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

CSV_NAMES=['MAIN_RESULTS','SOURCE_RESULTS','SOURCE_EQUAL_RESULTS','SOURCE_SD',
    'INTERVALS','PRIMARY_FAMILY_INTERVALS','SEED_RESULTS','SOURCE_SEED_RESULTS',
    'POOLED_SEED_RESULTS','SEED_MEAN_SD','SOURCE_SEED_SD',
    'TRAIN_CELLS','VALIDATION_REPLAY','COVERAGE']
SUMMARY_KEYS=['train_seed','arm','held_index','held_source','epoch','epochs',
    'threshold','rule','parameters','initial_sha256','checkpoint_sha256',
    'preprocessor_sha256','seconds','reused','validation_sha256']

def install_route(common,orchestrator):
    assert common.csv_out.__globals__ is common.__dict__
    def export(name,rows):
        old=common.PUB
        try:
            common.PUB=orchestrator.GLOBAL_PUB
            common.csv_out(name,rows)
        finally:common.PUB=old
    orchestrator.export_csv=export
    return export

def read_rows(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def seed43_rows(rows):
    assert len(rows)==60
    assert {int(r['train_seed']) for r in rows}=={42,43,44}
    result=[{k:v for k,v in r.items() if k!='train_seed'} for r in rows if int(r['train_seed'])==43]
    assert len(result)==20 and len({(r['arm'],r['held_source']) for r in result})==20
    return result

def snapshots(C,M):
    fr=C.jr(C.BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json')
    seal=C.jr(C.BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json')
    assert fr['status']=='ALL60_FROZEN_BEFORE_NEW_TEST' and seal['status']=='ALL60_SEALED_BEFORE_METRICS'
    assert fr['source']==seal['source']==C.sources()
    result={'global_freeze':C.sha(C.BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json'),
            'global_seal':C.sha(C.BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json')}
    for seed in [42,43,44]:
        root=M.seed_root(seed)
        assert C.sha(root/'FROZEN_BEFORE_TEST_PRIVATE.json')==fr['records'][str(seed)]
        assert C.sha(root/'TEST_SCORE_SEAL_PRIVATE.json')==seal['records'][str(seed)]
        choice=C.jr(root/'FROZEN_BEFORE_TEST_PRIVATE.json');scores=C.jr(root/'TEST_SCORE_SEAL_PRIVATE.json')
        assert len(choice['cells'])==len(scores['files'])==20
        for r in choice['cells']:
            key=f"{seed}|{r['arm']}|{r['held_index']}"
            for field in ['checkpoint','validation']:
                value=C.sha(r[field]);assert value==r[field+'_sha256']
                result[key+'|'+field]=value
        for name,h in scores['files'].items():
            assert C.sha(root/'scores_private'/(name+'.pkl'))==h
            result[f'{seed}|score|{name}']=h
    return result

def repair_existing(C,M):
    misplaced=C.ROOT/'public';target=M.GLOBAL_PUB
    backup=C.BASE_ROOT/'reporting_io_failure_preserved_v1'
    if not backup.exists():
        backup.mkdir()
        for n in CSV_NAMES:
            source=misplaced/(n+'.csv');assert source.is_file(),n
            shutil.copy2(source,backup/source.name)
        for n in ['REPORT_STATUS.json','REPORT.md']:
            shutil.copy2(target/n,backup/n)
        original={p.name:C.sha(p) for p in backup.iterdir() if p.is_file()}
        C.write(backup/'ORIGINAL_AGGREGATE_HASHES.json',original)
    original=C.jr(backup/'ORIGINAL_AGGREGATE_HASHES.json')
    for n,h in original.items():assert C.sha(backup/n)==h
    # Move no scientific artifacts. Copy the exact already-computed anonymous
    # tables to the intended directory; numeric values stay byte-identical.
    for n in CSV_NAMES:
        source=backup/(n+'.csv');dest=target/(n+'.csv')
        if dest.exists():assert C.sha(dest)==C.sha(source),('Unexplained existing global table',n)
        else:shutil.copy2(source,dest)
    # The first bad routing also placed combined60 rows over three per-seed43
    # public summaries. Recover metadata from the immutable selection ledger
    # and actual preserved replay/coverage rows; never invent a new validation.
    selected=C.jr(C.ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json')['cells']
    assert len(selected)==20
    C.csv_out('TRAIN_CELLS',[{k:r[k] for k in SUMMARY_KEYS} for r in selected])
    for name in ['VALIDATION_REPLAY','COVERAGE']:
        C.csv_out(name,seed43_rows(read_rows(backup/(name+'.csv'))))
    return {n:original[n+'.csv'] for n in CSV_NAMES if n not in ['TRAIN_CELLS','VALIDATION_REPLAY','COVERAGE']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['repair','report_verify_package']);args=parser.parse_args()
    assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
    os.environ['LOCO_TRAIN_SEED']='43'
    import psutil
    psutil.Process().cpu_affinity(list(range(16,32)))
    code=Path(r'D:\drechan_loco_five_models_3seed_v1')
    sys.path.insert(0,str(code))
    import common as C
    import orchestrate as M
    M.init()
    proof=C.jr(C.BASE_ROOT/'seed43/public/ADMISSION.json')
    assert len(proof['source'])==12 and C.sources()==proof['source']
    metric_hash=C.sha(C.BASE_ROOT/'ALL_PATIENT_METRICS_PRIVATE.pkl') if args.stage=='repair' else None
    before=snapshots(C,M)
    if args.stage=='repair':unchanged=repair_existing(C,M)
    else:unchanged={}
    install_route(C,M)
    if args.stage=='report_verify_package':M.report()
    M.verify()
    after=snapshots(C,M);assert before==after
    if metric_hash:assert C.sha(C.BASE_ROOT/'ALL_PATIENT_METRICS_PRIVATE.pkl')==metric_hash
    assert C.sources()==proof['source']
    for name,h in unchanged.items():assert C.sha(M.GLOBAL_PUB/(name+'.csv'))==h
    check=dict(status='PASS',scope='REPORTING_OUTPUT_ROUTING_ONLY',frozen_scientific_files=12,
               all_scientific_source_hashes_unchanged=True,all60_checkpoint_validation_score_bindings_unchanged=True,
               original_computed_metric_tables_byte_identical=(args.stage=='repair'),no_training_or_inference_rerun=True,
               no_threshold_or_model_change=True,artifact_fingerprint_sha256=C.digest(before),
               helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               original_failure='csv_out resolves common.PUB, not pipeline.PUB; combined rows overwrote3public seed43 summaries',
               repair='Preserve all misplaced outputs, copy exact numeric tables, restore20seed43 metadata rows, route CSV to common.PUB temporarily, independent verification')
    C.write(M.GLOBAL_PUB/'REPORTING_IO_REPAIR_PUBLIC.json',check)
    M.package()
    print('REPORTING_IO_ONLY_REPAIR_INDEPENDENT_VERIFICATION_AND_PACKAGE_PASS',flush=True)

if __name__=='__main__':main()
