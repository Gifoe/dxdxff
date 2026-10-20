"""Immutable original roles/math, complete three-seed execution and audited reuse."""
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
ROOT = Path(r'D:\drechan_f0_complete_ablation_3seed_v1_runtime')
PUB = ROOT/'public'
PARENT = Path(r'D:\drechan_f0_relative_guidance_pr_seed42_v1')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


current = module('complete3_current42_common', PARENT/'common.py')
legacy_parent = current.parent
sys.path.insert(0, str(HERE))
MODEL = module('complete3_model', HERE/'model.py')
feature_math = module('complete3_feature_pr', HERE/'feature_pr.py')
io, legacy, EV = current.io, current.legacy, current.EV
sha, read, save, write, freeze, jr, digest = io.sha, io.read, io.save, io.write, io.freeze, io.jr, io.digest
atomic_torch, state_hash, seed_all = io.atomic_torch, io.state_hash, io.seed_all
probability, METRICS = current.probability, current.METRICS
ARMS, SEEDS = MODEL.ARMS, [42, 43, 44]
FILES = ['PROTOCOL_LOCK.json', 'relative_guidance.py', 'feature_pr.py', 'model.py', 'common.py',
         'test_math.py', 'test_model.py', 'test_reporting.py', 'admit.py', 'train.py', 'verify_internal.py',
         'evaluate_outer.py', 'report_outer.py', 'verify_outer.py', 'run_all.ps1', 'affinity_entry.py', 'package_public.py']
CONTEXT = {}
sys.modules['common'] = OWN
sys.modules['model'] = MODEL


def sources():
    return {n: sha(HERE/n) for n in FILES}


def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True)
    PUB.mkdir(exist_ok=True)
    try:
        previous, core, fac, records, rec, banks = current.setup(False)
    finally:
        sys.modules['common'] = OWN
        sys.modules['model'] = MODEL
        sys.path.insert(0, str(HERE))
    lock = jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms'] == ARMS and lock['seeds'] == SEEDS and lock['parameters'] == MODEL.PARAMETERS
    for source, key in [(current.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json', 'parent_freeze_sha256'),
                        (current.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json', 'parent_score_seal_sha256'),
                        (legacy_parent.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json', 'legacy_freeze_sha256'),
                        (legacy_parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json', 'legacy_score_seal_sha256')]:
        assert sha(source) == lock[key]
    assert jr(current.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status'] == 'PASS'
    current_freeze, legacy_freeze = current.frozen(), legacy_parent.frozen()
    assert rec['R14'].shape == (7635, 14) and len(set(rec['meta']['patient'])) == 80
    assert sum(len(q['records']) for q in rec['ledgers']) == 256
    binding = dict(source=sources(), parent_source=current.sources(), parent_input=digest(previous),
                   parent_freeze=lock['parent_freeze_sha256'], parent_seal=lock['parent_score_seal_sha256'],
                   legacy_freeze=lock['legacy_freeze_sha256'], legacy_seal=lock['legacy_score_seal_sha256'],
                   reconstruction=io.EXPECTED_REC, raw_registry=io.EXPECTED_REG,
                   torch=torch.__version__, numpy=np.__version__, scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:
        freeze(ROOT/'INPUT_FREEZE_PRIVATE.json', binding)
    CONTEXT.update(binding=binding, rec=rec, current_freeze=current_freeze, legacy_freeze=legacy_freeze, maps={})
    return binding, core, fac, records, rec, banks


def bank(fold, banks, arm):
    assert arm in ARMS and CONTEXT
    if arm in ['B0_ORIGINAL', 'RAW_ONLY']:
        return current.bank(fold, banks, 'B0_ORIGINAL')
    if arm in ['WO_FEATURE_PR', 'WO_BOTH_PR']:
        return current.bank(fold, banks, 'WO_BOTH_PR')
    full = current.bank(fold, banks, 'FULL')
    if arm not in MODEL.DROP:
        return full
    path = ROOT/'banks_private'/digest(CONTEXT['binding'])[:16]/f'{arm}_fold{fold}.pkl'
    if path.exists():
        stored = read(path)
        assert stored['binding'] == CONTEXT['binding']
        value = stored['bank']
    else:
        raw = CONTEXT['rec']['R14'][full['archive_indices']].copy()
        j = MODEL.DROP[arm]
        raw[:, 2*j:2*j+2] = 0.
        x, pre = feature_math.prepare(raw, full['patient'], full['train'])
        active = [k for k in range(14) if k not in [2*j, 2*j+1]]
        assert np.array_equal(x[:, active], full['x'][:, active])
        assert np.array_equal(x[:, 2*j:2*j+2], np.zeros_like(x[:, 2*j:2*j+2]))
        value = dict(full, x=x, pre=pre)
        save(path, dict(binding=CONTEXT['binding'], bank=value))
    for key in ['patient', 'channel', 'y', 'train', 'val', 'archive_indices']:
        assert np.array_equal(value[key], full[key])
    assert value['roles'] == full['roles'] and set(value['pre']['fit_patient_ids']) == value['roles']['fit']
    assert set(value['patient']).isdisjoint(value['roles']['test']) and np.isfinite(value['x']).all()
    return value


def query_features(raw, patients, pre, arm, core):
    raw = np.asarray(raw).copy()
    if arm in ['B0_ORIGINAL', 'RAW_ONLY']:
        z = core.patient_z(raw, patients)
    elif arm in ['WO_FEATURE_PR', 'WO_BOTH_PR']:
        z = raw
    else:
        if arm in MODEL.DROP:
            j = MODEL.DROP[arm]
            raw[:, 2*j:2*j+2] = 0.
        return feature_math.query(raw, patients, pre)
    return ((np.where(np.isnan(z), pre['imputer_statistics'], z)-pre['mean'])/pre['scale']).astype(np.float32)


def scale(fold):
    return current.scale(fold)


def preprocessor(d, ss):
    return current.preprocessor(d, ss)


def source_for(core, fac, records, fold, seed):
    assert seed in SEEDS and 1 <= fold <= 5
    s = seed+1009*fold
    core.seed_all(s)
    template = legacy.OLD.P3.MorphCorrection(legacy.OLD.CompactDelta(fac(s).network, records)).cuda()
    source = legacy_parent.old.MODEL.SingleHead('F0_AFTER_GELU', template.cnn, template.gate,
                                               type(template).features, s+170003).cuda()
    del template
    return source


def make(core, fac, records, fold, arm, seed):
    assert arm in ARMS
    source = source_for(core, fac, records, fold, seed)
    model = MODEL.CompleteF0(source, arm).cuda()
    del source
    return model


def reference(arm, seed, fold, optional=False):
    if seed != 42:
        return None
    if arm in current.ARMS:
        fr, original_arm, kind = CONTEXT['current_freeze'], arm, 'CURRENT42'
    elif arm == 'RAW_ONLY':
        if not optional:
            admission = jr(PUB/'ADMISSION.json')
            if not admission['raw_reuse_by_fold'][str(fold)]:
                return None
        fr, original_arm, kind = CONTEXT['legacy_freeze'], 'RAW_ONLY', 'LEGACY_RAW42'
    else:
        return None
    r = next(r for r in fr['cells'] if r['arm'] == original_arm and r['fold'] == fold)
    assert sha(r['checkpoint']) == r['checkpoint_sha256'] and sha(r['validation']) == r['validation_sha256']
    state, val = torch.load(r['checkpoint'], map_location='cpu', weights_only=False), read(r['validation'])
    assert state['binding'] == val['binding']
    r = dict(r, arm=arm, seed=seed, epochs=len(val['history']), reused=True, reuse_kind=kind,
             source_arm=original_arm, parameters=MODEL.PARAMETERS[arm])
    return r, state, val


def reference_score(r):
    root = current.ROOT if r['reuse_kind'] == 'CURRENT42' else legacy_parent.ROOT
    key = f"{r['source_arm']}_{r['fold']}"
    path = root/'scores_private'/(key+'.pkl')
    assert sha(path) == jr(root/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][key]
    q = read(path)
    assert q['checkpoint'] == r['checkpoint_sha256'] and q['threshold'] == r['threshold']
    return q, sha(path), root


def store():
    result = legacy.OLD.P3.Inputs(legacy.OLD.BASELINE, Path(r'D:\dgm_net_seed42_v1_runtime'), ROOT)
    result.arm = 'M1_SCALED_DGM'
    return result


def mismatch(mask, channels, pid, fold):
    key = (fold, pid)
    cb = dict(input=digest(CONTEXT['binding']), channels=digest(channels),
              mask=legacy.array_sha(mask.cpu().numpy()), mapping_salt=42)
    path = ROOT/'mismatch_maps_private'/digest(CONTEXT['binding'])[:16]/f'fold{fold}'/(hashlib.sha256(pid.encode()).hexdigest()+'.pkl')
    if key in CONTEXT['maps']:
        q, ph = CONTEXT['maps'][key]
        assert q['binding'] == cb
    else:
        if path.exists():
            q = read(path)
            assert q['binding'] == cb
        else:
            mapping, info = MODEL.channel_mapping(mask, channels, pid, fold)
            q = dict(binding=cb, permutation=mapping.cpu(), info=info)
            save(path, q)
        ph = sha(path)
        CONTEXT['maps'][key] = (q, ph)
    assert q['info']['total_valid_windows'] == int(mask.sum())
    audit = {('mismatch_'+k): q['info'][k] for k in ['changed_valid_windows', 'eligible_valid_windows', 'singleton_valid_windows']}
    return q['permutation'].to(mask.device), dict(audit, mapping_sha256=ph)


def forward(model, d, ix, inputs, ss, fold, binding):
    assert len(set(d['patient'][ix])) == 1
    x = torch.as_tensor(d['x'][ix], device='cuda')
    if model.branch == 'phys':
        return model(x)['logit'], dict(valid_windows=0)
    pid = str(d['patient'][ix[0]])
    channels = d['channel'][ix].astype(str).tolist()
    raw, mask, times, delta, _ = inputs.get(pid, channels, 'cuda')
    permutation, audit = None, {}
    if model.arm == 'EARLY_FILM_MISMATCH':
        permutation, audit = mismatch(mask, channels, pid, fold)
    output = model(x, raw, mask, times, delta, ss, permutation)['logit']
    info = dict(model.last_guidance_audit)
    if 'singleton_valid_windows' in info:
        info['relative_singleton_valid_windows'] = info.pop('singleton_valid_windows')
    if 'valid_windows' in info:
        assert info['valid_windows'] == int(mask.sum())
    info.update(audit)
    info['valid_windows'] = int(mask.sum())
    return output, info


def predict(model, d, ix, inputs, ss, fold, binding):
    model.eval()
    result = np.empty(len(ix), np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj = np.flatnonzero(d['patient'][ix] == pid)
            logits, _ = forward(model, d, ix[jj], inputs, ss, fold, binding)
            result[jj] = logits.cpu().numpy()
    assert np.isfinite(result).all()
    return result


def expected_cells():
    return {(a, s, f) for a in ARMS for s in SEEDS for f in range(1, 6)}


def key_for(arm, seed, fold):
    return f'{arm}_seed{seed}_fold{fold}'


def csv_out(name, rows):
    assert rows and all(set(r) == set(rows[0]) for r in rows)
    path = PUB/(name+'.csv')
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def frozen():
    value = jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert value['source'] == sources() and value['input'] == jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'], r['seed'], r['fold']) for r in value['cells']} == expected_cells()
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status'] == 'PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json') == value['internal_verification']
    for r in value['cells']:
        assert sha(r['checkpoint']) == r['checkpoint_sha256'] and sha(r['validation']) == r['validation_sha256']
    return value


def vector(rows):
    return np.asarray([[r[m] for m in METRICS] for r in rows], np.float64)


def interval(values, draw, levels=(.025, .975)):
    assert values.shape == (80, 9) and np.isfinite(values).all()
    samples = np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0, 10000, 250)])
    return values.mean(0), np.quantile(samples, levels, axis=0)


PAIRS = {'FULL_MINUS_'+a: {'FULL': 1, a: -1} for a in ARMS if a != 'FULL'}
INTERACTIONS = {
    'PR_INTERACTION': {'FULL': 1, 'WO_FEATURE_PR': -1, 'WO_RAW_PR': -1, 'WO_BOTH_PR': 1},
    'GUIDANCE_INTERACTION': {'FULL': 1, 'WO_EARLY_FILM': -1, 'WO_LATE_GATE': -1, 'WO_BOTH_GUIDANCE': 1}}
