"""Label-blind raw cache preparation and immutable D0/D2 control replay."""
import argparse
import gc
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_raw import ROOT, RAW, EXPORT, norm
from common import METRICS, metric, sha, write_json
from model import SourceA0
from sampling import normalize_record, permutation

PRIOR = Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
CORE = Path(r'E:\DRE-nips\new-pipeline\7-11\pr_uncertainty_aware_supervision_seed42_v1\code')


def load_controls(protocol, manifest):
    lock = json.loads(protocol.read_text())
    expected = json.loads(manifest.read_text())
    assert sha(CORE / 'uas_core.py') == lock['dataset']['core_sha256']
    sys.path.insert(0, str(CORE))
    from uas_core import PRMLP
    banks, rows, gates = {}, [], []
    for fold in range(1, 6):
        paths = [f'fold{fold}/BANK_PRIVATE.pt', f'fold{fold}/D2_FROZEN_PRIVATE.pt',
                 f'fold{fold}/D2/BEST_PRIVATE.pt']
        for rel in paths:
            assert sha(PRIOR / rel) == expected[rel.replace('/', '\\')]
        d = torch.load(PRIOR / paths[0], weights_only=False, map_location='cpu')
        assert set(d['patient'][d['train']]).isdisjoint(d['patient'][d['val']])
        d2 = torch.load(PRIOR / paths[1], weights_only=False, map_location='cpu')
        state = torch.load(PRIOR / paths[2], weights_only=False, map_location='cpu')
        assert d2['checkpoint_sha256'] == sha(PRIOR / paths[2])
        old = Path(d['A0_checkpoint'])
        orig = torch.load(old, weights_only=False, map_location='cpu')
        m0 = PRMLP().eval()
        m0.load_state_dict(orig['model'])
        m2 = SourceA0().eval()
        m2.load_state_dict(state['model'])
        with torch.no_grad():
            l0 = m0(torch.as_tensor(d['x'])).numpy()
            l2 = m2(torch.as_tensor(d['x']), torch.as_tensor(d['g'])).numpy()
        va = d['val']
        drift0 = float(np.max(np.abs(torch.sigmoid(torch.from_numpy(l0[va])).numpy()-d['a0_scores'][va])))
        drift2 = float(np.max(np.abs(l2[va]-d2['logits'][va])))
        assert drift0 < 2e-7 and drift2 < 2e-6
        for p in sorted(set(d['patient'][va])):
            ix = va[d['patient'][va] == p]
            ix = ix[np.argsort(d['channel'][ix].astype(str), kind='stable')]
            for arm, score, tau in [('D0', d['a0_scores'], d['A0_threshold']),
                                    ('D2', d2['scores'], d2['threshold'])]:
                rows.append({'fold': fold, 'patient': p, 'center': str(d['center'][ix[0]]),
                             'method': arm, **metric(d['y'][ix], score[ix], tau)})
        gates.append({'fold': fold, 'D0_checkpoint_sha256': sha(old),
                      'D2_checkpoint_sha256': sha(PRIOR/paths[2]), 'bank_sha256': sha(PRIOR/paths[0]),
                      'D0_score_drift': drift0, 'D2_logit_drift': drift2,
                      'D0_threshold': d['A0_threshold'], 'D2_threshold': d2['threshold']})
        banks[fold] = d
    frame = pd.DataFrame(rows)
    assert len(frame) == 130 and frame.patient.nunique() == 47
    values = {}
    for arm, target in [('D0', .6380797828499001), ('D2', .6479260773776448)]:
        g = frame[frame.method == arm]
        assert len(g) == 65 and g.channels.sum() == 6273
        assert abs(g.macro_f1.mean()-target) < 1e-12
        values[arm] = g[METRICS].mean().to_dict()
    torch.save(rows, ROOT / 'CONTROL_METRICS_PRIVATE.pt')
    write_json(ROOT/'audit/D0_D2_REPRODUCTION.json', {
        'status': 'PASS', 'exact_frozen_scores_after_checkpoint_gate': True,
        'controls_retrained': False, 'metrics': values, 'folds': gates})
    return banks


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--manifest', type=Path, required=True)
    args = a.parse_args()
    torch.set_num_threads(2)
    gate = json.loads((ROOT/'audit/RAW_ALIGNMENT_AUDIT.json').read_text())
    assert gate['status'] == 'PASS'
    print('CONTROL_REPLAY_START', flush=True)
    load_controls(args.protocol, args.manifest)
    print('D0_D2_REPLAY_PASS', flush=True)
    with np.load(EXPORT, allow_pickle=True) as d:
        patients, channels = d['patient'].astype(str), np.array([norm(c) for c in d['channel']])
    with (ROOT/'RAW_LEDGER_PRIVATE.pkl').open('rb') as f:
        ledger = pickle.load(f)
    assert ledger['raw_hash'] == gate['raw_sha256']
    print('LOADING_RAW_CACHE', flush=True)
    with RAW.open('rb') as f:
        raw = pickle.load(f)
    records = defaultdict(list)
    for r in raw['run_records']:
        records[(str(r['subject_id']), str(r['run_id']))].append(r)
    bypatient = defaultdict(list)
    for row in ledger['records']:
        bypatient[row['patient']].append(row)
    directory = ROOT/'raw_windows_private'
    directory.mkdir(exist_ok=True)
    binding = {'raw': gate['raw_sha256'], 'feature': gate['feature_cache_sha256'],
               'protocol': sha(args.protocol), 'sampling_code': sha(Path(__file__).with_name('sampling.py'))}
    index, stats = {}, []
    for number, p in enumerate(sorted(bypatient)):
        dest = directory/f'patient_{number:03d}.pt'
        names = channels[patients == p]
        runs = bypatient[p]
        C, S, W = len(names), len(runs), max(len(r['times']) for r in runs)
        if dest.exists():
            obj = torch.load(dest, weights_only=False, map_location='cpu')
            assert obj['binding'] == binding and np.array_equal(obj['channels'], names)
        else:
            wave = np.zeros((C, S, W, 512), np.float32)
            aux = np.zeros((C, S, W, 2), np.float32)
            mask = np.zeros((C, S, W), bool)
            coordinates = np.zeros((S, W), np.float32)
            lookup = {c: i for i, c in enumerate(names)}
            for s, row in enumerate(runs):
                rr = [r for r in records[(p, row['run'])]
                      if str(r['sample'].get('sample_id', r['run_id'])) == row['sample']
                      and tuple(norm(c) for c in r['channel_names_norm']) == row['names']]
                assert len(rr) == 1
                source = np.asarray(rr[0]['sample']['raw_waveform'])
                w, ad = normalize_record(source, row['valid_start'], row['valid_samples'],
                                         row['starts'], row['valid'], row['fs'])
                ix = np.array([lookup[c] for c in row['names']])
                n = len(row['times'])
                wave[ix, s, :n] = w
                aux[ix, s, :n] = ad
                mask[ix, s, :n] = row['valid']
                coordinates[s, :n] = row['times']
            perm = permutation(mask.any(-1), p)
            obj = {'binding': binding, 'patient': p, 'channels': names, 'wave': torch.from_numpy(wave),
                   'aux': torch.from_numpy(aux), 'mask': torch.from_numpy(mask),
                   'times': coordinates, 'runs': [r['run'] for r in runs],
                   'counts': [len(r['times']) for r in runs], 'perm': perm}
            temp = dest.with_suffix('.partial')
            torch.save(obj, temp)
            temp.replace(dest)
        index[p] = {'file': str(dest), 'sha256': sha(dest)}
        stats.append({'channels': C, 'seizures': S, 'max_candidates': W,
                      'valid_channels': int(obj['mask'].any(-1).any(-1).sum()),
                      'shuffled_channels': int((obj['perm'] != np.arange(C)).sum())})
        print('RAW_WINDOWS_PREPARED', number+1, '/80', flush=True)
    with (ROOT/'RAW_INDEX_PRIVATE.json').open('w', encoding='utf-8') as f:
        json.dump({'binding': binding, 'patients': index}, f)
    write_json(ROOT/'audit/RAW_NORMALIZATION_AUDIT.json', {
        'status': 'PASS', 'patients': 80, 'context': 'measured record interval median/MAD',
        'population_parameters': False, 'labels_used': False, 'finite_windows': True,
        'resampling': 'actual fs -> 256Hz; scipy resample_poly Kaiser5; exactly512',
        'frequency_information_created': False, 'cache_binding': binding})
    write_json(ROOT/'audit/RAW_CHANNEL_SHUFFLE_AUDIT.json', {
        'status': 'PASS', 'within_patient_only': True, 'labels_used': False,
        'seizure_membership_and_validity_preserved': True, 'patients': 80,
        'channels': sum(r['channels'] for r in stats),
        'shuffled_channels': sum(r['shuffled_channels'] for r in stats),
        'unshufflable_channels': sum(r['channels']-r['shuffled_channels'] for r in stats),
        'patients_no_shuffle': sum(r['shuffled_channels']==0 for r in stats),
        'policy': 'stable cyclic derangement within identical valid-seizure signatures'})
    write_json(ROOT/'audit/WINDOW_SAMPLING_AUDIT.json', {
        'status': 'CANDIDATE_CACHE_READY', 'max_seizures_per_patient': max(r['seizures'] for r in stats),
        'candidate_cache_channels': sum(r['channels'] for r in stats),
        'label_blind': True, 'invalid_windows_not_encoded': True,
        'sparse_selection_test_required': True})
    print('RAW_CACHE_AND_CONTROL_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
