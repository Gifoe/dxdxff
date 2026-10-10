"""D2-only frozen outer audit. No training, selection, or raw signal I/O.

prepare: replay FIT+VAL and seal all historical dependencies without TEST labels.
evaluate: infer once, seal all prediction files, then read clinical labels.
Only the explicit public aggregate allowlist may leave the private runtime.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_transform(x, patient, pre, core):
    z = core.patient_z(x, patient)
    stats = np.asarray(pre['imputer_statistics'])
    assert np.isfinite(stats).all() and (np.asarray(pre['scale']) > 0).all()
    imputed = np.where(np.isnan(z), stats, z)
    result = ((imputed - pre['mean']) / pre['scale']).astype(np.float32)
    assert result.shape == x.shape and np.isfinite(result).all()
    return result


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temp)
    temp.replace(path)


def read_raw(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files if k != 'y'}


def role_ids(raw, fold):
    take = raw['split_fold'] == fold
    ids = {r: set(raw['split_patient'][take & (raw['split_role'] == r)])
           for r in ['fit', 'validation', 'test']}
    assert all(ids[a].isdisjoint(ids[b]) for a, b in
               [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')])
    return ids


def dependencies(a, lock):
    source = a.source
    paths = {'features': a.features, 'split': a.split,
             'original_protocol': source / 'a0_source_posterior_decoder_seed42_v1/PROTOCOL_LOCK.json',
             'core': source / 'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py',
             'model_source': source / 'a0_source_posterior_decoder_seed42_v1/code/model.py',
             'metric_source': source / 'a0_source_posterior_decoder_seed42_v1/code/common.py'}
    expected = {'features': lock['feature_sha256'], 'split': lock['split_sha256'],
                **{k: lock[k + '_sha256'] for k in
                   ['original_protocol', 'core', 'model_source', 'metric_source']}}
    for k, path in paths.items():
        assert sha(path) == expected[k], k
    for fold in range(1, 6):
        old = a.original / f'fold{fold}'
        for k, rel, ref in [('checkpoint', 'D2/BEST_PRIVATE.pt', lock['checkpoint_sha256'][fold-1]),
                            ('bank', 'BANK_PRIVATE.pt', lock['bank_sha256'][fold-1])]:
            path = old / rel
            assert sha(path) == ref, (k, fold)
            paths[f'{k}_{fold}'] = path
        paths[f'frozen_development_{fold}'] = old / 'D2_FROZEN_PRIVATE.pt'
        paths[f'formal_complete_{fold}'] = old / 'FORMAL_COMPLETE.json'
    paths['historical_tests'] = a.original / 'public/DECODER_UNIT_TEST_AUDIT.json'
    paths['amendment'] = a.amendment
    paths['outer_code'] = Path(__file__)
    paths['contract_tests'] = Path(__file__).parent.parent / 'tests/test_contract.py'
    return paths


def infer(model_module, ck, x, g):
    model = model_module.SourceA0().eval()
    model.load_state_dict(ck['model'], strict=True)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x), torch.from_numpy(g)).numpy()
        scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    assert np.isfinite(logits).all() and np.isfinite(scores).all()
    return logits, scores


def prepare(a, lock, paths, core, model, common):
    assert not (a.runtime / 'FROZEN_BEFORE_OUTER.json').exists(), 'Never replace freeze'
    assert not (a.runtime / 'OUTER_PREDICTIONS_SEAL.json').exists()
    raw = read_raw(a.features)  # Never load the y member in preparation.
    assert raw['x'].shape == (7635, 88) and len(set(raw['patient'])) == 80
    assert len(set(zip(raw['patient'], raw['channel']))) == 7635
    order_hash = hashlib.sha256(json.dumps(raw['features'].tolist(), separators=(',', ':')).encode()).hexdigest()
    assert order_hash == lock['feature_order_sha256']
    split = pd.read_csv(a.split)
    split_rows = set(zip(split.outer_fold, split.split_role.astype(str), split.subject_id.astype(str)))
    assert split_rows == set(zip(raw['split_fold'], raw['split_role'].astype(str), raw['split_patient'].astype(str)))
    tests = json.loads(paths['historical_tests'].read_text())
    assert tests['status'] == 'PASS'
    rows, all_test, total_channels = [], [], 0
    for fold in range(1, 6):
        bank = torch.load(paths[f'bank_{fold}'], map_location='cpu', weights_only=False)
        ck = torch.load(paths[f'checkpoint_{fold}'], map_location='cpu', weights_only=False)
        ref = torch.load(paths[f'frozen_development_{fold}'], map_location='cpu', weights_only=False)
        marker = json.loads(paths[f'formal_complete_{fold}'].read_text())
        ids = role_ids(raw, fold)
        di = np.flatnonzero(np.isin(raw['patient'], list(ids['fit'] | ids['validation'])))
        ti = np.flatnonzero(np.isin(raw['patient'], list(ids['test'])))
        for k in ['patient', 'channel', 'center', 'features']:
            assert np.array_equal(raw[k] if k == 'features' else raw[k][di], bank[k])
        assert set(bank['patient'][bank['train']]) == ids['fit']
        assert set(bank['patient'][bank['val']]) == ids['validation']
        assert set(bank['pre']['fit_patient_ids']) == ids['fit']
        assert set(bank['patient']).isdisjoint(ids['test'])
        assert bank['binding']['core'] == lock['core_sha256']
        assert bank['binding']['protocol'] == lock['original_protocol_sha256']
        for filename, refhash in bank['binding']['code'].items():
            assert sha(a.source / 'a0_source_posterior_decoder_seed42_v1/code' / filename) == refhash
        assert bank['binding']['code'] == tests['code_sha256']
        assert ref['checkpoint_sha256'] == sha(paths[f'checkpoint_{fold}']) == marker['checkpoint_sha256']
        assert ref['binding'] == ck['binding']['source'] == marker['binding']
        assert ck['binding']['seed'] == 42 + 1009 * fold
        assert ck['binding']['penalty'] == .001 and ck['binding']['max_epochs'] == 30
        original_bind = common.digest({'bank': sha(paths[f'bank_{fold}']),
            'protocol': lock['original_protocol_sha256'], 'tests': sha(paths['historical_tests']),
            'code': bank['binding']['code']})
        assert original_bind == ck['binding']['source']
        assert ck['threshold'] == ref['threshold'] == marker['threshold'] == lock['nez_thresholds'][fold-1]
        assert ck['epoch'] == ref['epoch'] == marker['epoch'] == lock['epochs'][fold-1]
        # Rebuild FIT-only moments solely as a parity control; inference uses stored pre.
        replay_x, replay_pre = core.prepare(raw['x'][di], bank['patient'], bank['train'])
        for k in ['imputer_statistics', 'mean', 'scale', 'var']:
            assert np.array_equal(replay_pre[k], bank['pre'][k]), (fold, k)
        frozen_x = frozen_transform(raw['x'][di], bank['patient'], bank['pre'], core)
        assert np.array_equal(frozen_x, replay_x) and np.array_equal(frozen_x, bank['x'])
        assert np.array_equal(model.source_index(bank['center']), bank['g'])
        logits, scores = infer(model, ck, frozen_x, bank['g'])
        ld = float(np.max(np.abs(logits-ref['logits'])))
        pdiff = float(np.max(np.abs(scores-ref['scores'])))
        assert ld <= lock['replay_logit_absolute_tolerance']
        assert pdiff <= lock['replay_probability_absolute_tolerance']
        assert len(ids['test']) == lock['expected_test_patients_by_fold'][fold-1]
        assert len(ti) == lock['expected_test_channels_by_fold'][fold-1]
        all_test.extend(ids['test']); total_channels += len(ti)
        pre_hash = common.digest({k: bank['pre'][k].tolist() for k in
                                  ['imputer_statistics', 'mean', 'scale', 'var']})
        rows.append({'fold': fold, 'epoch': ck['epoch'], 'nez_threshold': ck['threshold'],
            'checkpoint_sha256': sha(paths[f'checkpoint_{fold}']), 'preprocessor_sha256': pre_hash,
            'fit_patients': len(ids['fit']), 'validation_patients': len(ids['validation']),
            'outer_patients': len(ids['test']), 'outer_channels': len(ti),
            'max_logit_replay_drift': ld, 'max_probability_replay_drift': pdiff,
            'frozen_preprocessor_bitwise_parity': True, 'outer_fit_overlap': 0,
            'test_label_access': False})
    assert len(all_test) == len(set(all_test)) == 80 and total_channels == 7635
    freeze = {
        'status': 'PASS', 'all_folds_locked_before_outer': True, 'outer_test_labels_loaded': False,
        'hashes': {k: sha(p) for k, p in paths.items()}, 'folds': rows,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'device': 'cpu', 'threads': 2, 'test_patients': 80, 'test_channels': 7635}
    common.write_json(a.runtime / 'FROZEN_BEFORE_OUTER.json', freeze)
    common.write_json(a.runtime / 'public/FROZEN_DEPENDENCY_AUDIT.json', freeze)
    print('ALL_FIVE_FOLDS_FROZEN_TEST_LABELS_UNREAD', flush=True)


def evaluate(a, lock, paths, core, model, common):
    marker = a.runtime / 'public/OUTER_STATUS.json'
    assert not marker.exists(), 'Completed scientific evaluation must not be rerun'
    freeze_path = a.runtime / 'FROZEN_BEFORE_OUTER.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['status'] == 'PASS'
    before = {k: sha(p) for k, p in paths.items()}
    assert before == freeze['hashes'], 'Freeze dependency changed'
    raw = read_raw(a.features)
    files, all_indices = {}, []
    binding = sha(freeze_path)
    for fold in range(1, 6):
        bank = torch.load(paths[f'bank_{fold}'], map_location='cpu', weights_only=False)
        ck = torch.load(paths[f'checkpoint_{fold}'], map_location='cpu', weights_only=False)
        ids = role_ids(raw, fold)
        ix = np.flatnonzero(np.isin(raw['patient'], list(ids['test'])))
        out = a.runtime / f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
        if out.exists():
            saved_hash = json.loads(out.with_suffix('.sha256.json').read_text())
            assert saved_hash == {'binding': binding, 'sha256': sha(out)}
            cell = torch.load(out, map_location='cpu', weights_only=False)
            assert cell['binding'] == binding and cell['fold'] == fold
            assert np.array_equal(cell['indices'], ix)
            assert 'y' not in cell and np.isfinite(cell['scores']).all()
        else:
            x = frozen_transform(raw['x'][ix], raw['patient'][ix], bank['pre'], core)
            g = model.source_index(raw['center'][ix]); assert (g >= 0).all()
            logits, scores = infer(model, ck, x, g)
            atomic_torch(out, {'binding': binding, 'fold': fold, 'indices': ix,
                'logits': logits, 'scores': scores, 'threshold': ck['threshold']})
            common.write_json(out.with_suffix('.sha256.json'), {'binding': binding, 'sha256': sha(out)})
        all_indices.extend(ix.tolist()); files[str(fold)] = sha(out)
        print('OUTER_PREDICTIONS_SEALED', fold, len(ix), flush=True)
    assert sorted(all_indices) == list(range(7635))
    seal_path = a.runtime / 'OUTER_PREDICTIONS_SEAL.json'
    seal = {'binding': binding, 'files': files, 'all_predictions_complete_before_y': True}
    if seal_path.exists():
        assert json.loads(seal_path.read_text()) == seal
    else:
        common.write_json(seal_path, seal)
    # First and only full-cohort clinical label access: after all five predictions sealed.
    with np.load(a.features, allow_pickle=True) as f:
        y = f['y']
    assert y.shape == (7635,) and set(y) == {0, 1}
    assert int((y == 0).sum()) == 1743
    rows, scores_all, decisions_all = [], np.empty(7635), np.empty(7635, dtype=bool)
    for fold in range(1, 6):
        path = a.runtime / f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
        assert sha(path) == files[str(fold)]
        cell = torch.load(path, map_location='cpu', weights_only=False)
        ix, scores, tau = cell['indices'], cell['scores'], cell['threshold']
        scores_all[ix] = scores; decisions_all[ix] = scores >= tau
        # Clinical labels already present in original FIT+VAL bank must match export exactly.
        bank = torch.load(paths[f'bank_{fold}'], map_location='cpu', weights_only=False)
        di = np.flatnonzero(np.isin(raw['patient'], bank['patient']))
        assert np.array_equal(y[di], bank['y'])
        for pat in sorted(set(raw['patient'][ix])):
            local = np.flatnonzero(raw['patient'][ix] == pat)
            local = local[np.argsort(raw['channel'][ix][local].astype(str), kind='stable')]
            si = ix[local]
            centers = set(raw['center'][si]); assert len(centers) == 1
            rows.append({'patient': pat, 'fold': fold, 'center': str(raw['center'][si[0]]),
                         **common.metric(y[si], scores[local], tau)})
    df = pd.DataFrame(rows)
    assert len(df) == df.patient.nunique() == 80 and int(df.channels.sum()) == 7635
    atomic_torch(a.runtime / 'PATIENT_METRICS_PRIVATE.pt', rows)
    pub = a.runtime / 'public'; pub.mkdir(exist_ok=True)
    bootstrap = common.cluster_bootstrap(df.patient, df[common.METRICS].to_numpy(), draws=10000)
    aggregate = []
    for j, key in enumerate(common.METRICS):
        samples = bootstrap[:, j]
        aggregate.append({'model': 'D2', 'unit': 'patient_equal_outer', 'metric': key,
            'estimate': float(df[key].mean()), 'ci_low': float(np.nanpercentile(samples, 2.5)),
            'ci_high': float(np.nanpercentile(samples, 97.5)), 'estimable_patients': int(df[key].notna().sum()),
            'total_patients': 80, 'channels': 7635, 'draws': 10000, 'seed': 42})
    pd.DataFrame(aggregate).to_csv(pub / 'OUTER_PATIENT_EQUAL_METRICS.csv', index=False)
    pd.DataFrame(bootstrap, columns=common.METRICS).assign(draw=np.arange(1,10001)).to_csv(
        a.runtime / 'OUTER_BOOTSTRAP_DRAWS_PRIVATE.csv', index=False)
    for key, filename in [('fold', 'OUTER_FOLD_METRICS.csv'), ('center', 'OUTER_CENTER_METRICS.csv')]:
        group_rows = []
        for value, group in df.groupby(key):
            group_rows.append({key: value, 'patients': len(group), 'channels': int(group.channels.sum()),
                'ez_channels': int(np.rint((group.channels * group.observed_ez_fraction).sum())),
                **group[common.METRICS].mean().to_dict(),
                **{m + '_estimable_patients': int(group[m].notna().sum()) for m in common.METRICS}})
        pd.DataFrame(group_rows).to_csv(pub / filename, index=False)
    # Secondary pooled unit: do not apply one fold's numeric threshold to other folds.
    canonical = np.lexsort((raw['channel'].astype(str), raw['patient'].astype(str)))
    pooled = common.metric(y[canonical], scores_all[canonical], 0., pred=decisions_all[canonical])
    pd.DataFrame([{'model': 'D2', 'unit': 'pooled_outer_channels', **pooled,
                   'mrr_top1_not_patient_local_do_not_use': True}]).drop(
        columns=['ez_mrr', 'top1_is_ez']).to_csv(pub / 'OUTER_POOLED_CHANNEL_METRICS.csv', index=False)
    after = {k: sha(p) for k, p in paths.items()}
    assert before == after == freeze['hashes']
    common.write_json(pub / 'IMMUTABILITY_AND_COVERAGE_AUDIT.json', {
        'status': 'PASS', 'all_dependencies_unchanged': True, 'freeze_sha256': binding,
        'prediction_seal_sha256': sha(seal_path), 'test_patients_unique_once': 80,
        'test_channels_unique_once': 7635, 'EZ_channels': 1743, 'NEZ_channels': 5892,
        'frozen_thresholds': lock['nez_thresholds'], 'all_scores_sealed_before_full_y_access': True,
        'FIT_VAL_export_label_parity': True, 'no_retraining': True, 'no_threshold_tuning': True,
        'no_raw_eeg_extraction': True, 'private_patient_predictions_not_published': True,
        'confusion_sum': {k: int(df[k].sum()) for k in ['TP','FP','TN','FN']}})
    common.write_json(marker, {'status': 'ONE_FROZEN_D2_OUTER_PASS_COMPLETE',
        'scientific_status': lock['scientific_status'], 'patients': 80, 'channels': 7635,
        'freeze_sha256': binding, 'primary_metrics': df[common.METRICS].mean().to_dict(),
        'public_artifacts': sorted(q.name for q in pub.iterdir() if q.is_file()) + ['OUTER_STATUS.json']})
    print('ONE_FROZEN_D2_OUTER_PASS_COMPLETE', json.dumps(df[common.METRICS].mean().to_dict()), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'evaluate'])
    p.add_argument('--runtime', type=Path, default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    p.add_argument('--original', type=Path, default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--source', type=Path, default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--features', type=Path, default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--split', type=Path, default=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))
    p.add_argument('--amendment', type=Path, default=Path(__file__).parent.parent / 'PROTOCOL_AMENDMENT.json')
    a = p.parse_args(); torch.set_num_threads(2)
    a.runtime.mkdir(parents=True, exist_ok=True)
    lock = json.loads(a.amendment.read_text())
    paths = dependencies(a, lock)
    core = load_module('historical_d2_core', paths['core'])
    model = load_module('historical_d2_model', paths['model_source'])
    common = load_module('historical_d2_metrics', paths['metric_source'])
    (prepare if a.phase == 'prepare' else evaluate)(a, lock, paths, core, model, common)


if __name__ == '__main__':
    main()
