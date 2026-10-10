"""Independently verify sealed scores and seed statistics; no model replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(r'C:\d2_ictal_multiseed_outer_v1_runtime')
    baseline = Path(r'C:\d2_ictal_outer_seed42_runtime')
    original = Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
    pub = root/'public'
    assert json.loads((pub/'OUTER_STATUS.json').read_text())['status'] == 'FIVE_SEED_D2_OUTER_COMPLETE'
    freeze = json.loads((root/'FROZEN_BEFORE_OUTER.json').read_text())
    seal = json.loads((root/'OUTER_PREDICTIONS_SEAL.json').read_text())
    assert seal['binding'] == sha(root/'FROZEN_BEFORE_OUTER.json') and len(seal['files']) == 25
    with np.load(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz', allow_pickle=True) as f:
        y, patient, channel = f['y'], f['patient'], f['channel']
    table = pd.DataFrame(torch.load(root/'PATIENT_METRICS_PRIVATE.pt', map_location='cpu', weights_only=False))
    pretrain = json.loads((pub/'PRETRAIN_AUDIT.json').read_text())
    output_rows = []; max_drift = 0.; new_epochs = 0
    for rec in freeze['selections']:
        seed, fold = rec['seed'], rec['fold']
        ckpath = (original/f'fold{fold}/D2' if seed == 42 else root/f'seed{seed}/fold{fold}/D2')/'BEST_PRIVATE.pt'
        assert sha(ckpath) == rec['checkpoint_sha256']
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        assert ck['binding']['initial_hash'] == pretrain['folds'][fold-1]['initial_hashes'][str(seed)]
        assert ck['binding']['seed'] == seed+1009*fold and ck['binding']['device'] == 'cuda'
        assert ck['binding']['penalty'] == .001 and ck['binding']['max_epochs'] == 30
        assert sum(t.numel() for t in ck['model'].values()) == 9221
        assert ck['threshold'] == rec['nez_threshold'] and ck['epoch'] == rec['epoch']
        assert sha(original/f'fold{fold}/BANK_PRIVATE.pt') == rec['bank_sha256']
        if seed != 42:
            lastpath = ckpath.parent/'LAST_PRIVATE.pt'
            done = json.loads((ckpath.parent/'COMPLETE_PRIVATE.json').read_text())
            assert sha(lastpath) == done['last_sha256'] and sha(ckpath) == done['best_sha256']
            new_epochs += torch.load(lastpath, map_location='cpu', weights_only=False)['epoch']
        path = baseline/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt' if seed == 42 else root/f'seed{seed}/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'
        assert sha(path) == seal['files'][f'{seed}_{fold}']
        cell = torch.load(path, map_location='cpu', weights_only=False)
        assert cell['threshold'] == ck['threshold']
        ix = cell['indices']; scores = cell['scores'].astype(float)
        for pat in sorted(set(patient[ix])):
            k = np.flatnonzero(patient[ix] == pat)
            k = k[np.argsort(channel[ix][k].astype(str), kind='stable')]
            a, s = y[ix[k]], scores[k]
            assert len(set(a)) == 2
            pred = (s >= cell['threshold']).astype(int)
            r0 = recall_score(a, pred, pos_label=0); r1 = recall_score(a, pred, pos_label=1)
            rank = np.argsort(s, kind='stable'); ez = np.flatnonzero(a[rank] == 0)
            values = {'macro_f1': f1_score(a, pred, labels=[0, 1], average='macro', zero_division=0),
                'ez_f1': f1_score(a, pred, pos_label=0, zero_division=0),
                'nez_f1': f1_score(a, pred, pos_label=1, zero_division=0),
                'balanced_accuracy': (r0+r1)/2, 'sensitivity': r0, 'specificity': r1,
                'accuracy': accuracy_score(a, pred), 'ez_auroc': roc_auc_score(1-a, 1-s),
                'ez_auprc': average_precision_score(1-a, 1-s),
                'ez_mrr': float(1/(ez[0]+1)), 'top1_is_ez': float(a[rank[0]] == 0)}
            recorded = table[(table.seed == seed) & (table.patient == pat)].iloc[0]
            for key, value in values.items():
                drift = abs(value-recorded[key]); max_drift = max(max_drift, drift)
                assert drift < 1e-12
            output_rows.append({'seed': seed, 'patient': pat, **values})
    independent = pd.DataFrame(output_rows)
    assert len(independent) == 400 and independent.patient.nunique() == 80
    summaries = pd.read_csv(pub/'MULTISEED_SUMMARY.csv')
    by_seed = pd.read_csv(pub/'OUTER_BY_SEED.csv')
    keys = summaries.metric.tolist()
    for seed, group in independent.groupby('seed'):
        assert len(group) == group.patient.nunique() == 80
        for key in keys:
            assert abs(group[key].mean()-by_seed[by_seed.seed == seed][key].iloc[0]) < 1e-12
    estimates = independent.groupby('seed')[keys].mean()
    patient_means = independent.groupby('patient')[keys].mean().to_numpy()
    ix = np.random.default_rng(42).integers(0, 80, size=(10000, 80))
    draws = patient_means[ix].mean(1)
    for j, row in summaries.iterrows():
        values = estimates[row.metric].to_numpy()
        assert abs(values.mean()-row['mean']) < 1e-12
        assert abs(values.std(ddof=1)-row['std']) < 1e-12
        assert abs(values.min()-row['minimum']) < 1e-12 and abs(values.max()-row['maximum']) < 1e-12
        assert abs(np.percentile(draws[:, j], 2.5)-row.patient_bootstrap_ci_low) < 1e-12
        assert abs(np.percentile(draws[:, j], 97.5)-row.patient_bootstrap_ci_high) < 1e-12
    output = pub/'INDEPENDENT_VERIFICATION.json'; assert not output.exists()
    output.write_text(json.dumps({'status': 'PASS', 'model_patient_cells': 400, 'unique_patients': 80,
        'metrics_per_cell': 11, 'maximum_metric_absolute_drift': max_drift, 'checkpoint_bindings_checked': 25,
        'new_training_cells': 20, 'new_training_epochs': new_epochs, 'seed42_cells_reused': 5,
        'seed_mean_and_sample_sd_independently_verified': True, 'bootstrap_draws': 10000,
        'patient_cluster_CI_of_seed_average_verified': True, 'no_prediction_replay': True,
        'no_threshold_reoptimization': True, 'source_sha256': sha(__file__)}, indent=2), encoding='utf-8')
    print('INDEPENDENT_MULTISEED_VERIFICATION_PASS', max_drift, new_epochs, flush=True)


if __name__ == '__main__': main()
