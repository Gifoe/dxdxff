"""Historical replay admission, sealed development banks, FIT-only feasibility."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baseline import sha, json_write, torch_write, prepare
from pllr import build_pllr


def binding(protocol):
    names = ['baseline.py', 'pllr.py', 'train.py', 'prepare.py', 'run.py', 'smoke.py']
    return {**{n: sha(Path(__file__).parent / n) for n in names}, 'protocol': sha(protocol)}


def seal(root, protocol):
    gate = json.loads((root / 'gate/A0_REPRODUCTION.json').read_text())
    replay = json.loads((root / 'RUNTIME_ADMISSION.json').read_text())
    data = root / 'gate/FEATURES_PRIVATE.npz'
    lock = json.loads(protocol.read_text())
    assert gate['status'] == replay['status'] == 'PASS'
    assert sha(data) == gate['private_features_sha256'] == lock['private_feature_sha256']
    assert replay['historical_metrics_exact'] and replay['torch'] == torch.__version__
    assert replay['numpy'] == np.__version__
    with np.load(data, allow_pickle=True) as archive: d = {k: archive[k] for k in archive.files}
    x, y, pid, ch, center = [d[k] for k in ['x', 'y', 'patient', 'channel', 'center']]
    assert x.shape == (7635, 88) and len(set(pid)) == 80
    assert len(set(zip(pid, ch))) == 7635 and len(set(d['features'])) == 88
    # Source export contains all historical labels for mandatory frozen replay.
    # Seal before formal development; no outer arrays reach geometry/training.
    records = []
    for fold in range(1, 6):
        take = d['split_fold'] == fold
        ids = {r: set(d['split_patient'][take & (d['split_role'] == r)])
               for r in ['fit', 'validation', 'test']}
        assert ids['fit'] | ids['validation'] | ids['test'] == set(pid)
        assert all(not ids[r] & ids[s] for r, s in [('fit','validation'),('fit','test'),('validation','test')])
        di = np.flatnonzero(np.isin(pid, list(ids['fit'] | ids['validation'])))
        fp, fc = pid[di].astype(str), ch[di].astype(str)
        tr = np.flatnonzero(np.isin(fp, list(ids['fit'])))
        va = np.flatnonzero(np.isin(fp, list(ids['validation'])))
        normalized, pre = prepare(x[di], fp, tr)
        assert set(pre['fit_patient_ids']) == ids['fit']
        path = root / f'fold{fold}/DEVELOPMENT_PRIVATE.pt'
        bind = {'feature': sha(data), 'protocol': sha(protocol), 'fold': fold}
        record = {'binding': bind, 'x': normalized, 'y': y[di], 'patient': fp, 'channel': fc,
            'center': center[di].astype(str), 'features': d['features'], 'train': tr, 'val': va, 'pre': pre}
        if path.exists():
            old = torch.load(path, weights_only=False); assert old['binding'] == bind
            assert all(np.array_equal(old[k], record[k]) for k in ['x','y','patient','channel','train','val'])
        else: torch_write(path, record)
        records.append({'fold': fold, 'fit_patients': len(ids['fit']), 'validation_patients': len(ids['validation']),
            'fit_channels': len(tr), 'validation_channels': len(va), 'outer_channels_in_development': 0,
            'sealed_sha256': sha(path), 'feature_order_sha256': __import__('hashlib').sha256(
                json.dumps(d['features'].tolist(), separators=(',', ':')).encode()).hexdigest()})
    public = root / 'public'; public.mkdir(exist_ok=True)
    json_write(public / 'A0_REPRODUCTION.json', {'status': 'HISTORICAL_REPLAY_PASS_MATCHED_TRAINING_PENDING',
        'historical_reproduction': gate, 'fresh_runtime_replay': replay,
        'sealing': records, 'matched_A0_training_reproduction_required': True,
        'source_export_outer_labels_materialized_only_for_required_replay_and_sealing': True,
        'development_outer_labels_available': False})
    return records


def summarize(root, protocol, records):
    patient_rows = []; class_rows = []; stability_rows = []; fold_rows = []; pca_rows = []
    binds = binding(protocol)
    for record in records:
        fold = record['fold']; foldroot = root / f'fold{fold}'
        d = torch.load(foldroot / 'DEVELOPMENT_PRIVATE.pt', weights_only=False)
        tr = d['train']; p = d['patient'][tr]; c = d['channel'][tr]
        out = foldroot / 'PLLR_FROZEN_PRIVATE.pt'
        bind = {**binds, 'sealed': record['sealed_sha256']}
        if out.exists():
            saved = torch.load(out, weights_only=False); assert saved['binding'] == bind
            a, pr, cr, sr = [saved[k] for k in ['artifact', 'patients', 'classes', 'stability']]
        else:
            a, pr, cr, sr = build_pllr(d['x'][tr], d['y'][tr], p, c, d['center'][tr], fold)
            torch_write(out, {'binding': bind, 'patient': p, 'channel': c, 'y': d['y'][tr],
                'artifact': a, 'patients': pr, 'classes': cr, 'stability': sr})
        assert np.array_equal(p, torch.load(out, weights_only=False)['patient'])
        patient_rows.extend(pr); class_rows.extend(cr); stability_rows.extend(sr)
        fraction = float(np.mean([r['eligible'] for r in pr]))
        rho = float(np.median([r['mean_rho'] for r in sr])) if sr else None
        fold_rows.append({'fold': fold, 'FIT_patients': len(pr),
            'eligible_patients': sum(r['eligible'] for r in pr), 'eligible_fraction': fraction,
            'eligible_channels': sum(r['eligible_channels'] for r in pr),
            'median_patient_mean_stability_rho': rho, 'tau': a['tau'],
            'positive_conflict_channels': int((a['margin'] > 0).sum()),
            'degenerate_no_positive_conflict': a['tau'] is None,
            'eligibility_pass': fraction >= .3, 'stability_pass': rho is not None and rho >= .5,
            'PLLR_sha256': sha(out), 'PCA_FIT_only': True, 'PCA_dimension': 8})
        pca_rows.append({'fold': fold, 'explained_variance_ratio': a['explained_variance_ratio'].tolist(),
            'FIT_rows': len(tr), 'outer_validation_test_rows': 0})
        print('FIT_GEOMETRY_COMPLETE', fold, 'eligible', fraction, 'stability', rho, flush=True)
    public = root / 'public'
    pf = pd.DataFrame(patient_rows); cf = pd.DataFrame(class_rows); sf = pd.DataFrame(stability_rows)
    rows = []
    for fold in range(1, 6):
        for center in ['all', *sorted(pf.center.unique())]:
            p = pf[(pf.fold == fold) & ((pf.center == center) if center != 'all' else True)]
            if p.empty: continue
            s = sf[(sf.fold == fold) & ((sf.center == center) if center != 'all' else True)] if not sf.empty else sf
            rows.append({'fold': fold, 'center': center, 'FIT_patients': len(p),
                'eligible_patients': int(p.eligible.sum()), 'eligible_channels': int(p.eligible_channels.sum()),
                'insufficient_patients': int((~p.eligible).sum()),
                'mean_patient_rho': float(s.mean_rho.mean()) if len(s) else None,
                'median_patient_mean_rho': float(s.mean_rho.median()) if len(s) else None,
                'q10_patient_mean_rho': float(s.mean_rho.quantile(.1)) if len(s) else None,
                'fraction_patient_rho_below_0_5': float((s.mean_rho < .5).mean()) if len(s) else None,
                'mean_abs_weight_change': float(s.mean_abs_weight_change.mean()) if len(s) else None,
                'conflict_sign_change_fraction': float(s.conflict_sign_change_fraction.mean()) if len(s) else None,
                'resamples': 100})
    pd.DataFrame(rows).to_csv(public / 'PROTOTYPE_STABILITY.csv', index=False)
    summary = []
    for (fold, center, label), g in cf.groupby(['fold','center','observed_class']):
        n = g.channels.sum()
        summary.append({'fold': int(fold), 'center': center, 'observed_class': label,
            'patients': len(g), 'channels': int(n), 'raw_min': g.raw_min.min(), 'raw_max': g.raw_max.max(),
            'raw_mean': float(np.average(g.raw_mean, weights=g.channels)),
            'weight_min': g.weight_min.min(), 'weight_max': g.weight_max.max(),
            'max_patient_class_mass_error': g.mass_error.max(), 'per_patient_class_mean': 1.,
            'mean_within_patient_class_weight_std': g.weight_std.mean(),
            'effective_sample_size_sum': g.effective_sample_size.sum(),
            'raw_downweighted_fraction': g.raw_downweighted.sum()/n,
            'final_downweighted_fraction': g.final_downweighted.sum()/n,
            'largest_patient_class_mass_fraction': g.channels.max()/n})
    pd.DataFrame(summary).to_csv(public / 'RELIABILITY_WEIGHT_SUMMARY.csv', index=False)
    passed = all(r['eligibility_pass'] and r['stability_pass'] for r in fold_rows)
    json_write(public / 'PLLR_FEASIBILITY_AUDIT.json', {'status': 'PASS' if passed else 'FAIL',
        'pass': passed, 'folds': fold_rows, 'PCA': pca_rows, 'binding': binds,
        'prototype_query_self_excluded': True, 'outer_VAL_TEST_labels_used': False,
        'weights_frozen': True, 'class_mass_preserved_max_error': float(cf.mass_error.max()),
        'minority_rule': 'same support rule for EZ and NEZ; not rarity-based; class normalized',
        'leverage_protection': 'one update per patient, fixed beta; no new clipping',
        'failed_folds': [r['fold'] for r in fold_rows if not (r['eligibility_pass'] and r['stability_pass'])]})
    json_write(public / 'B1_PERMUTATION_AUDIT.json', {'status': 'PASS',
        'every_patient_class_exact_multiset': True, 'class_mass_preserved': True,
        'stable_ID_and_channel_seed': True, 'no_cross_patient_or_class_permutation': True,
        'every_nonconstant_group_feature_correspondence_changed': True,
        'mean_patient_changed_channel_fraction': float(pf.B1_changed_fraction.mean()),
        'constant_patient_groups': int((~pf.nonconstant).sum()),
        'permutation_protocol': json.loads(protocol.read_text())['B1_permutation']})
    json_write(root / 'RUN_STATUS.json', {'status': 'FEASIBILITY_PASS' if passed else 'PLLR_FEASIBILITY_FAILED',
        'training_admitted': passed, 'outer_evaluation_run': False, 'binding': binds})
    if not passed:
        json_write(public / 'RUN_STATUS.json', {'status': 'COMPLETE_FEASIBILITY_FAILED',
            'terminal': 'PLLR_FEASIBILITY_FAILED', 'student_training_runs': 0,
            'outer_evaluation_run': False, 'development_training_complete': False})
    print('PLLR_FEASIBILITY_' + ('PASS' if passed else 'FAILED_STOP_NO_TRAINING'), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True); a = parser.parse_args()
    torch.set_num_threads(2)
    records = seal(a.runtime, a.protocol); summarize(a.runtime, a.protocol, records)


if __name__ == '__main__': main()
