"""Exact original0.6538 roles and body; derived physiology preprocessing only."""
import csv
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

OWN = sys.modules[__name__]
HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\drechan_f0_relative_guidance_pr_seed42_v1_runtime')
PUB = ROOT/'public'
PARENT = Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


parent = module('relative_guidance_parent_common', PARENT/'common.py')
sys.path.insert(0, str(HERE))
MODEL = module('relative_guidance_current_model', HERE/'model.py')
from feature_pr import prepare as prepare_feature_pr, query as query_feature_pr
io, legacy, EV = parent.io, parent.legacy, parent.EV
sha, read, save, write, freeze, jr, digest = io.sha, io.read, io.save, io.write, io.freeze, io.jr, io.digest
atomic_torch, state_hash, seed_all = io.atomic_torch, io.state_hash, io.seed_all
probability, METRICS = parent.probability, parent.METRICS
ARMS = MODEL.ARMS
REFERENCES = {'B0_ORIGINAL': 'FULL_F0'}
REF_NAMES = REFERENCES
PREVROOT = parent.ROOT
NEW = [a for a in ARMS if a not in REFERENCES]
CONTRASTS = {
    'FULL_MINUS_B0': {'FULL': 1, 'B0_ORIGINAL': -1},
    'FEATURE_PR': {'FULL': 1, 'WO_FEATURE_PR': -1},
    'RAW_PR': {'FULL': 1, 'WO_RAW_PR': -1},
    'BOTH_PR': {'FULL': 1, 'WO_BOTH_PR': -1},
    'WINDOW_PR': {'FULL': 1, 'WO_WINDOW_PR': -1},
    'EARLY_FILM': {'FULL': 1, 'WO_EARLY_FILM': -1},
    'LATE_GATE': {'FULL': 1, 'WO_LATE_GATE': -1},
    'PRE_REFERENCE': {'FULL': 1, 'WO_PRE_REFERENCE': -1},
    'PR_INTERACTION': {'FULL': 1, 'WO_FEATURE_PR': -1, 'WO_RAW_PR': -1, 'WO_BOTH_PR': 1},
}
PRIMARY = [name for name in CONTRASTS if name not in ['BOTH_PR', 'PR_INTERACTION']]
FILES = ['PROTOCOL_LOCK.json', 'relative_guidance.py', 'feature_pr.py', 'model.py', 'common.py',
         'test_math.py', 'test_model.py', 'test_reporting.py', 'admit.py', 'train.py', 'verify_internal.py',
         'evaluate_outer.py', 'report_outer.py', 'verify_outer.py', 'run_all.ps1',
         'affinity_entry.py', 'package_public.py']
CONTEXT = {}
sys.modules['common'] = OWN
sys.modules['model'] = MODEL


def sources():
    return {name: sha(HERE/name) for name in FILES}


def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True)
    PUB.mkdir(exist_ok=True)
    try:
        previous, core, fac, records, rec, banks = parent.setup(False)
    finally:
        sys.modules['common'] = OWN
        sys.modules['model'] = MODEL
        sys.path.insert(0, str(HERE))
    protocol = jr(HERE/'PROTOCOL_LOCK.json')
    assert protocol['arms'] == ARMS and protocol['parameters'] == MODEL.PARAMETERS
    assert sha(parent.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json') == protocol['parent_freeze_sha256']
    assert sha(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json') == protocol['parent_score_seal_sha256']
    assert jr(parent.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status'] == 'PASS'
    parent.frozen()
    assert rec['R14'].shape == (7635, 14) and len(set(rec['meta']['patient'])) == 80
    assert sum(len(ledger['records']) for ledger in rec['ledgers']) == 256
    binding = dict(source=sources(), parent_source=parent.sources(), parent_input=digest(previous),
                   parent_freeze=protocol['parent_freeze_sha256'], parent_seal=protocol['parent_score_seal_sha256'],
                   reconstruction=io.EXPECTED_REC, raw_registry=io.EXPECTED_REG,
                   torch=torch.__version__, numpy=np.__version__, scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:
        freeze(ROOT/'INPUT_FREEZE_PRIVATE.json', binding)
    CONTEXT.update(rec=rec, binding=binding)
    return binding, core, fac, records, rec, banks


def bank(fold, banks, arm):
    assert arm in ARMS and CONTEXT
    original = parent.bank(fold, banks, 'FULL_F0')
    if arm == 'B0_ORIGINAL':
        return original
    no_pr = parent.bank(fold, banks, 'WO_BOTH_PR')
    for key in ['patient', 'channel', 'y', 'train', 'val', 'archive_indices']:
        assert np.array_equal(original[key], no_pr[key])
    assert original['roles'] == no_pr['roles']
    if arm in ['WO_FEATURE_PR', 'WO_BOTH_PR']:
        return no_pr
    path = ROOT/'banks_private'/digest(CONTEXT['binding'])[:16]/f'fold{fold}.pkl'
    if path.exists():
        stored = read(path)
        assert stored['binding'] == CONTEXT['binding']
        result = stored['bank']
    else:
        raw = CONTEXT['rec']['R14'][original['archive_indices']]
        x, pre = prepare_feature_pr(raw, original['patient'], original['train'])
        assert np.array_equal(pre['FIT_unit_scale'], no_pr['pre']['scale'])
        assert np.array_equal(pre['FIT_unit_variance'], no_pr['pre']['var'])
        result = dict(original, x=x, pre=pre)
        save(path, dict(binding=CONTEXT['binding'], bank=result))
    for key in ['patient', 'channel', 'y', 'train', 'val', 'archive_indices']:
        assert np.array_equal(result[key], original[key])
    assert result['roles'] == original['roles']
    assert set(result['pre']['fit_patient_ids']) == result['roles']['fit']
    assert set(result['patient']).isdisjoint(result['roles']['test'])
    assert result['x'].shape == original['x'].shape and np.isfinite(result['x']).all()
    return result


def scale(fold):
    return parent.scale(fold)


def preprocessor(d, ss):
    return parent.preprocessor(d, ss)


def make(core, fac, records, fold, arm):
    source = parent.old.make(core, fac, records, fold, 'F0_AFTER_GELU')
    model = MODEL.RelativeF0(source, arm).cuda()
    del source
    return model


def reference(arm, fold):
    assert arm == 'B0_ORIGINAL'
    original = next(r for r in parent.frozen()['cells'] if r['arm'] == 'FULL_F0' and r['fold'] == fold)
    assert sha(original['checkpoint']) == original['checkpoint_sha256']
    assert sha(original['validation']) == original['validation_sha256']
    state = torch.load(original['checkpoint'], map_location='cpu', weights_only=False)
    validation = read(original['validation'])
    assert state['binding'] == validation['binding']
    return dict(original, arm=arm, reused=True, reference_arm='FULL_F0'), state, validation


def store():
    result = legacy.OLD.P3.Inputs(legacy.OLD.BASELINE, Path(r'D:\dgm_net_seed42_v1_runtime'), ROOT)
    result.arm = 'M1_SCALED_DGM'
    return result


def forward(model, d, ix, inputs, ss, fold, binding):
    assert len(set(d['patient'][ix])) == 1, 'A PR update requires one complete patient'
    pid = str(d['patient'][ix[0]])
    channels = d['channel'][ix].astype(str).tolist()
    raw, mask, times, delta, _ = inputs.get(pid, channels, 'cuda')
    x = torch.as_tensor(d['x'][ix], device='cuda')
    output = model(x, raw, mask, times, delta, ss)['logit']
    info = dict(model.last_guidance_audit)
    if 'valid_windows' in info:
        assert info['valid_windows'] == int(mask.sum())
    info['valid_windows'] = int(mask.sum())
    return output, info


def predict(model, d, ix, inputs, ss, fold, binding):
    model.eval()
    output = np.empty(len(ix), np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            positions = np.flatnonzero(d['patient'][ix] == pid)
            logits, _ = forward(model, d, ix[positions], inputs, ss, fold, binding)
            output[positions] = logits.cpu().numpy()
    assert np.isfinite(output).all()
    return output


def csv_out(name, rows):
    assert rows and all(set(row) == set(rows[0]) for row in rows)
    path = PUB/(name+'.csv')
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def frozen():
    value = jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert value['source'] == sources() and value['input'] == jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert len(value['cells']) == 45 and jr(PUB/'INTERNAL_VERIFICATION.json')['status'] == 'PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json') == value['internal_verification']
    for row in value['cells']:
        assert sha(row['checkpoint']) == row['checkpoint_sha256']
        assert sha(row['validation']) == row['validation_sha256']
    return value


def vector(rows):
    return np.array([[row[metric] for metric in METRICS] for row in rows], dtype=np.float64)


def interval(values, draw, levels=(.025, .975)):
    assert values.shape == (80, 9) and np.isfinite(values).all()
    samples = np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0, 10000, 250)])
    return values.mean(0), np.quantile(samples, levels, axis=0)
