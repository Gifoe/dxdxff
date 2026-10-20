"""Separate diagnostic namespace; never write to completed parent artifacts."""
import csv
import gc
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\drechan_dualview_traincal_diagnostic_v1_runtime'); PUB = ROOT/'public'
PARENT = Path(r'D:\drechan_raw_dualview_3seed_v1')
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('tcdiag_completed_parent', PARENT/'common.py')
rdv = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rdv; spec.loader.exec_module(rdv)
sys.path.insert(0, str(HERE))
sha, read, save, write, jr, freeze, digest = rdv.sha, rdv.read, rdv.save, rdv.write, rdv.jr, rdv.freeze, rdv.digest
FILES = ['PROTOCOL_LOCK.json','audit_math.py','audit_support.py','audit.py','test_audit.py','verify.py','run_all.ps1','affinity_entry.py']
ARMS = ['B0','PRUNED_REF','FULL']; LATENT = ['PRUNED_REF','FULL']; SEEDS = [42,43,44]


def sources(): return {n: sha(HERE/n) for n in FILES}


def setup():
    ROOT.mkdir(exist_ok=True); PUB.mkdir(exist_ok=True)
    binding, core, fac, records, rec, banks = rdv.setup(False)
    lock = jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['calibration_bodies'] == ARMS and lock['latent_bodies'] == LATENT
    assert sha(rdv.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json') == lock['parent_freeze_sha256']
    assert sha(rdv.ROOT/'SCORE_SEAL_PRIVATE.json') == lock['parent_seal_sha256']
    assert jr(rdv.PUB/'INDEPENDENT_VERIFICATION.json')['status'] == 'PASS'
    fr = rdv.frozen()
    own = dict(source=sources(), parent_input=digest(binding), parent_freeze=lock['parent_freeze_sha256'],
               parent_seal=lock['parent_seal_sha256'], mode=lock['mode'], torch=torch.__version__, numpy=np.__version__)
    freeze(ROOT/'INPUT_BINDING_PRIVATE.json', own)
    return own, core, fac, records, rec, banks, fr


def inputs():
    q = rdv.legacy.OLD.P3.Inputs(rdv.legacy.OLD.BASELINE, Path(r'D:\dgm_net_seed42_v1_runtime'), ROOT)
    q.arm = 'M1_SCALED_DGM'; return q


def expected(): return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}


def cell_path(arm, seed, fold): return ROOT/'cells_private'/f'{arm}_seed{seed}_fold{fold}.pkl'


def cell_binding(binding, r, d):
    return dict(input=digest(binding), arm=r['arm'], seed=r['seed'], fold=r['fold'],
                checkpoint=r['checkpoint_sha256'], calibration=r['calibration_sha256'],
                TRAIN_indices=rdv.legacy.array_sha(d['archive_indices'][d['train']]),
                CAL_indices=rdv.legacy.array_sha(d['archive_indices'][d['val']]),
                TRAIN_patients=digest(sorted(d['roles']['train'])), CAL_patients=digest(sorted(d['roles']['cal'])))


def csv_out(name, rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path = PUB/(name+'.csv'); temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def table(name):
    with (PUB/(name+'.csv')).open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
