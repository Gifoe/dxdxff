"""Original PR D2, four new seeds plus immutable seed42; no raw signal I/O."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def summarize(frame, metrics):
    assert frame.seed.nunique() == len(frame) and len(frame) >= 2
    return pd.DataFrame([{'metric': k, 'mean': frame[k].mean(),
        'std': frame[k].std(ddof=1), 'minimum': frame[k].min(),
        'maximum': frame[k].max(), 'n_seeds': len(frame), 'std_ddof': 1}
        for k in metrics])


def raw_without_y(a):
    with np.load(a.features, allow_pickle=True) as f:
        raw = {k: f[k] for k in f.files if k != 'y'}
    assert raw['x'].shape == (7635, 88) and len(set(raw['patient'])) == 80
    assert len(set(zip(raw['patient'], raw['channel']))) == 7635
    return raw


def roles(raw, fold):
    ix = raw['split_fold'] == fold
    ids = {r: set(raw['split_patient'][ix & (raw['split_role'] == r)])
           for r in ['fit', 'validation', 'test']}
    assert all(ids[x].isdisjoint(ids[y]) for x, y in
               [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')])
    return ids


def setup(a):
    lock = json.loads(a.protocol.read_text())
    assert lock['seeds'] == [0, 1, 2, 3, 42] and lock['new_training_seeds'] == [0, 1, 2, 3]
    original_code = a.source/'a0_source_posterior_decoder_seed42_v1/code'
    core_path = a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    common = module('multi_common', original_code/'common.py')
    amendment = a.source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    assert common.sha(amendment) == lock['baseline_amendment_sha256']
    ref = json.loads(amendment.read_text())
    assert common.sha(core_path) == ref['core_sha256']
    sys.path.insert(0, str(core_path.parent)); sys.path.insert(0, str(original_code))
    import uas_core, model, train
    code_hashes = {p.name: common.sha(p) for p in original_code.glob('*.py')}
    tests = json.loads((a.original/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())
    assert tests['status'] == 'PASS' and code_hashes == tests['code_sha256']
    assert common.sha(a.features) == ref['feature_sha256']
    split = Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert common.sha(split) == ref['split_sha256']
    assert common.sha(a.baseline/'FROZEN_BEFORE_OUTER.json') == lock['baseline_freeze_sha256']
    assert common.sha(a.baseline/'OUTER_PREDICTIONS_SEAL.json') == lock['baseline_prediction_seal_sha256']
    assert json.loads((a.baseline/'public/OUTER_STATUS.json').read_text())['status'] == 'ONE_FROZEN_D2_OUTER_PASS_COMPLETE'
    helper_path = a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py'
    helper = module('multi_outer_helper', helper_path)
    paths = {'features': a.features, 'split': split, 'protocol': a.protocol,
        'core': core_path, 'amendment': amendment, 'outer_helper': helper_path,
        'runner': Path(__file__), 'tests': Path(__file__).parents[1]/'tests/test_contract.py',
        'baseline_freeze': a.baseline/'FROZEN_BEFORE_OUTER.json',
        'baseline_seal': a.baseline/'OUTER_PREDICTIONS_SEAL.json'}
    for fold in range(1, 6):
        b = a.original/f'fold{fold}/BANK_PRIVATE.pt'
        c = a.original/f'fold{fold}/D2/BEST_PRIVATE.pt'
        assert common.sha(b) == ref['bank_sha256'][fold-1]
        assert common.sha(c) == ref['checkpoint_sha256'][fold-1]
        paths[f'original_bank_{fold}'] = b
        paths[f'original_checkpoint_{fold}'] = c
    binding = {'dependencies': {k: common.sha(v) for k, v in paths.items()},
               'original_training_code': code_hashes}
    a.runtime.mkdir(parents=True, exist_ok=True)
    return lock, ref, common, uas_core, model, train, helper, binding


def prepare(a, objects):
    lock, ref, common, core, model, trainer, helper, binding = objects
    pub = a.runtime/'public'; pub.mkdir(exist_ok=True)
    if (pub/'PRETRAIN_AUDIT.json').exists():
        assert json.loads((pub/'PRETRAIN_AUDIT.json').read_text())['binding'] == binding
        return
    raw = raw_without_y(a)
    assert hashlib.sha256(json.dumps(raw['features'].tolist(), separators=(',', ':')).encode()).hexdigest() == ref['feature_order_sha256']
    rows = []; all_test = []
    for fold in range(1, 6):
        d = torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt', map_location='cpu', weights_only=False)
        ids = roles(raw, fold)
        di = np.flatnonzero(np.isin(raw['patient'], list(ids['fit'] | ids['validation'])))
        for k in ['patient', 'channel', 'center']:
            assert np.array_equal(raw[k][di], d[k])
        assert np.array_equal(raw['x'][di], d['raw_x'], equal_nan=True)
        assert set(d['patient'][d['train']]) == ids['fit']
        assert set(d['patient'][d['val']]) == ids['validation']
        assert set(d['pre']['fit_patient_ids']) == ids['fit']
        assert not set(d['patient']) & ids['test']
        x, pre = core.prepare(d['raw_x'], d['patient'], d['train'])
        for k in ['imputer_statistics', 'mean', 'scale', 'var']:
            assert np.array_equal(pre[k], d['pre'][k])
        assert np.array_equal(x, d['x'])
        assert np.array_equal(helper.frozen_transform(d['raw_x'], d['patient'], d['pre'], core), d['x'])
        assert np.array_equal(model.source_index(d['center']), d['g'])
        assert len(ids['test']) == ref['expected_test_patients_by_fold'][fold-1]
        all_test.extend(ids['test'])
        initial_hashes = {}
        for seed in lock['seeds']:
            fold_seed = seed + 1009*fold
            core.seed_all(fold_seed)
            initial_hashes[str(seed)] = core.state_hash(model.SourceA0(fold_seed).state_dict())
        assert len(set(initial_hashes.values())) == 5
        original = torch.load(a.original/f'fold{fold}/D2/BEST_PRIVATE.pt', map_location='cpu', weights_only=False)
        assert original['binding']['initial_hash'] == initial_hashes['42']
        rows.append({'fold': fold, 'FIT_patients': len(ids['fit']), 'validation_patients': len(ids['validation']),
            'outer_patients': len(ids['test']), 'FIT_VAL_only': True, 'patient_z_retained': True,
            'stored_preprocessing_bitwise_parity': True, 'initial_hashes': initial_hashes})
    assert len(all_test) == len(set(all_test)) == 80
    common.write_json(pub/'PRETRAIN_AUDIT.json', {'status': 'PASS', 'binding': binding,
        'folds': rows, 'full_cohort_y_loaded': False, 'new_seeds_predeclared': lock['new_training_seeds']})
    print('PRETRAIN_PASS_FIVE_SEEDS_FIXED_SPLITS', flush=True)


def train_all(a, objects):
    lock, ref, common, core, model, trainer, helper, binding = objects
    audit = json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text())
    assert audit['binding'] == binding
    completed = 0
    for seed in lock['new_training_seeds']:
        for fold in range(1, 6):
            bankpath = a.original/f'fold{fold}/BANK_PRIVATE.pt'
            d = torch.load(bankpath, map_location='cpu', weights_only=False)
            cell = a.runtime/f'seed{seed}/fold{fold}/D2'
            trainer.train(d['x'], d['y'], d['patient'], d['g'], d['train'], d['val'], cell,
                seed+1009*fold, common.digest({'bank': common.sha(bankpath), 'binding': binding}), device='cuda')
            ck = torch.load(cell/'BEST_PRIVATE.pt', map_location='cpu', weights_only=False)
            assert ck['binding']['initial_hash'] == audit['folds'][fold-1]['initial_hashes'][str(seed)]
            completed += 1
            common.write_json(a.runtime/'public/TRAIN_PROGRESS.json', {'status': 'RUNNING',
                'new_cells_complete': completed, 'new_cells_total': 20, 'last_seed': seed, 'last_fold': fold})
            print('SEED_FOLD_SELECTED', seed, fold, ck['epoch'], ck['threshold'], flush=True)
    common.write_json(a.runtime/'public/TRAIN_PROGRESS.json', {'status': 'ALL_20_NEW_TRAIN_VALIDATION_COMPLETE',
        'new_cells_complete': 20, 'seed42_cells_reused': 5, 'torch_version': torch.__version__, 'outer_evaluated': False})


def checkpoint(a, seed, fold):
    return (a.original/f'fold{fold}/D2' if seed == 42 else a.runtime/f'seed{seed}/fold{fold}/D2')/'BEST_PRIVATE.pt'


def freeze(a, objects):
    lock, ref, common, core, model, trainer, helper, binding = objects
    out = a.runtime/'FROZEN_BEFORE_OUTER.json'
    if out.exists():
        assert json.loads(out.read_text())['binding'] == binding
        return
    assert json.loads((a.runtime/'public/TRAIN_PROGRESS.json').read_text())['new_cells_complete'] == 20
    rows = []
    for seed in lock['seeds']:
        for fold in range(1, 6):
            path = checkpoint(a, seed, fold)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['binding']['seed'] == seed+1009*fold
            assert ck['binding']['penalty'] == .001 and ck['binding']['max_epochs'] == 30
            assert ck['binding']['device'] == 'cuda'
            assert sum(t.numel() for t in ck['model'].values()) == 9221
            if seed != 42:
                done = json.loads((path.parent/'COMPLETE_PRIVATE.json').read_text())
                assert done['best_sha256'] == common.sha(path)
                assert done['selected_epoch'] == ck['epoch'] and done['threshold'] == ck['threshold']
            else:
                assert ck['epoch'] == ref['epochs'][fold-1] and ck['threshold'] == ref['nez_thresholds'][fold-1]
            rows.append({'seed': seed, 'fold': fold, 'fold_seed': seed+1009*fold, 'epoch': ck['epoch'],
                'nez_threshold': ck['threshold'], 'checkpoint_sha256': common.sha(path),
                'bank_sha256': common.sha(a.original/f'fold{fold}/BANK_PRIVATE.pt'), 'reused_seed42': seed == 42})
    common.write_json(out, {'status': 'PASS', 'binding': binding, 'selections': rows,
        'new_outer_inference_started': False, 'all_25_models_locked': True, 'new_full_cohort_y_loaded': False})
    common.write_json(a.runtime/'public/FREEZE_AUDIT.json', {'status': 'PASS', 'selections': rows,
        'freeze_sha256': common.sha(out), 'all_25_models_locked_before_new_outer': True})
    print('ALL_25_SELECTIONS_FROZEN', flush=True)


def evaluate(a, objects):
    lock, ref, common, core, model, trainer, helper, binding = objects
    pub = a.runtime/'public'
    assert not (pub/'OUTER_STATUS.json').exists(), 'Do not rerun completed scientific evaluation'
    frozen = json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text())
    assert frozen['binding'] == binding
    freeze_sha = common.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
    raw = raw_without_y(a); cells = {}; sealed = {}
    old_seal = json.loads((a.baseline/'OUTER_PREDICTIONS_SEAL.json').read_text())
    for seed in lock['seeds']:
        visited = []
        for fold in range(1, 6):
            rec = next(r for r in frozen['selections'] if r['seed'] == seed and r['fold'] == fold)
            assert common.sha(checkpoint(a, seed, fold)) == rec['checkpoint_sha256']
            assert common.sha(a.original/f'fold{fold}/BANK_PRIVATE.pt') == rec['bank_sha256']
            ix = np.flatnonzero(np.isin(raw['patient'], list(roles(raw, fold)['test'])))
            if seed == 42:
                path = a.baseline/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
                assert common.sha(path) == old_seal['files'][str(fold)]
                cell = torch.load(path, map_location='cpu', weights_only=False)
                assert np.array_equal(cell['indices'], ix) and cell['threshold'] == rec['nez_threshold']
            else:
                path = a.runtime/f'seed{seed}/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'
                if path.exists():
                    assert json.loads(path.with_suffix('.sha256.json').read_text()) == {'binding': freeze_sha, 'sha256': common.sha(path)}
                    cell = torch.load(path, map_location='cpu', weights_only=False)
                    assert cell['binding'] == freeze_sha and np.array_equal(cell['indices'], ix)
                else:
                    d = torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt', map_location='cpu', weights_only=False)
                    ck = torch.load(checkpoint(a, seed, fold), map_location='cpu', weights_only=False)
                    x = helper.frozen_transform(raw['x'][ix], raw['patient'][ix], d['pre'], core)
                    tm = model.SourceA0().eval(); tm.load_state_dict(ck['model'], strict=True)
                    with torch.inference_mode():
                        logits = tm(torch.from_numpy(x), torch.from_numpy(model.source_index(raw['center'][ix]))).numpy()
                        score = torch.sigmoid(torch.from_numpy(logits)).numpy()
                    cell = {'binding': freeze_sha, 'indices': ix, 'scores': score, 'logits': logits, 'threshold': ck['threshold']}
                    core.torch_write(path, cell)
                    common.write_json(path.with_suffix('.sha256.json'), {'binding': freeze_sha, 'sha256': common.sha(path)})
            assert np.isfinite(cell['scores']).all() and cell['threshold'] == rec['nez_threshold']
            cells[(seed, fold)] = cell; sealed[f'{seed}_{fold}'] = common.sha(path); visited.extend(ix.tolist())
        assert sorted(visited) == list(range(7635))
    common.write_json(a.runtime/'OUTER_PREDICTIONS_SEAL.json', {'binding': freeze_sha, 'files': sealed,
        'all_25_predictions_sealed_before_full_y': True, 'seed42_reused_not_recomputed': True})
    with np.load(a.features, allow_pickle=True) as f: y = f['y']
    assert len(y) == 7635 and int((y == 0).sum()) == 1743
    rows = []
    for (seed, fold), cell in cells.items():
        ix = cell['indices']
        for pat in sorted(set(raw['patient'][ix])):
            k = np.flatnonzero(raw['patient'][ix] == pat)
            k = k[np.argsort(raw['channel'][ix][k].astype(str), kind='stable')]
            si = ix[k]
            rows.append({'seed': seed, 'fold': fold, 'patient': pat, 'center': str(raw['center'][si[0]]),
                **common.metric(y[si], cell['scores'][k], cell['threshold'])})
    df = pd.DataFrame(rows)
    assert len(df) == 400 and df.patient.nunique() == 80
    for seed, group in df.groupby('seed'):
        assert len(group) == group.patient.nunique() == 80 and group.channels.sum() == 7635
    core.torch_write(a.runtime/'PATIENT_METRICS_PRIVATE.pt', rows)
    per_seed = df.groupby('seed')[common.METRICS].mean().reset_index()
    original_metrics = json.loads((a.baseline/'public/OUTER_STATUS.json').read_text())['primary_metrics']
    for k in common.METRICS:
        assert abs(per_seed[per_seed.seed == 42][k].iloc[0]-original_metrics[k]) < 1e-12
    per_seed.to_csv(pub/'OUTER_BY_SEED.csv', index=False)
    summary = summarize(per_seed, common.METRICS)
    patient_mean = df.groupby('patient')[common.METRICS].mean()
    draws = common.cluster_bootstrap(patient_mean.index, patient_mean.to_numpy(), 10000)
    summary['patient_bootstrap_ci_low'] = np.nanpercentile(draws, 2.5, axis=0)
    summary['patient_bootstrap_ci_high'] = np.nanpercentile(draws, 97.5, axis=0)
    summary['unique_patients'] = 80; summary['channels_per_seed'] = 7635
    summary.to_csv(pub/'MULTISEED_SUMMARY.csv', index=False)
    for field, filename in [('fold', 'OUTER_BY_SEED_FOLD.csv'), ('center', 'OUTER_BY_SEED_CENTER.csv')]:
        output = []
        for (seed, value), group in df.groupby(['seed', field]):
            output.append({'seed': seed, field: value, 'patients': len(group), 'channels': int(group.channels.sum()),
                **group[common.METRICS].mean().to_dict()})
        pd.DataFrame(output).to_csv(pub/filename, index=False)
    histories = []
    for seed in lock['new_training_seeds']:
        for fold in range(1, 6):
            last = torch.load(checkpoint(a, seed, fold).parent/'LAST_PRIVATE.pt', map_location='cpu', weights_only=False)
            histories.extend({'seed': seed, 'fold': fold, **h} for h in last['history'])
    pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv', index=False)
    pd.DataFrame(frozen['selections']).to_csv(pub/'MODEL_SELECTION.csv', index=False)
    assert setup(a)[-1] == binding
    common.write_json(pub/'IMMUTABILITY_AUDIT.json', {'status': 'PASS', 'historical_inputs_and_model_files_unchanged': True,
        'seed42_predictions_unchanged': True, 'unique_patients_per_seed': 80, 'channels_per_seed': 7635,
        'all_seeds_25_selections_before_outer': True, 'all_predictions_before_full_y': True,
        'freeze_sha256': freeze_sha, 'prediction_seal_sha256': common.sha(a.runtime/'OUTER_PREDICTIONS_SEAL.json')})
    common.write_json(pub/'OUTER_STATUS.json', {'status': 'FIVE_SEED_D2_OUTER_COMPLETE',
        'scientific_status': lock['scientific_status'], 'seeds': lock['seeds'], 'new_training_cells': 20,
        'reused_seed42_cells': 5, 'unique_patients': 80, 'channels_per_seed': 7635,
        'not_an_ensemble': True, 'not_400_independent_patients': True})
    print('FIVE_SEED_D2_OUTER_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'train', 'freeze', 'evaluate', 'all'])
    p.add_argument('--runtime', type=Path, default=Path(r'C:\d2_ictal_multiseed_outer_v1_runtime'))
    p.add_argument('--original', type=Path, default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--baseline', type=Path, default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    p.add_argument('--source', type=Path, default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--features', type=Path, default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--protocol', type=Path, default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    a = p.parse_args(); torch.set_num_threads(2); objects = setup(a); start = time.monotonic()
    phases = {'prepare': prepare, 'train': train_all, 'freeze': freeze, 'evaluate': evaluate}
    for phase in (phases if a.phase == 'all' else [a.phase]): phases[phase](a, objects)
    print('ELAPSED_SECONDS', time.monotonic()-start, flush=True)


if __name__ == '__main__': main()
