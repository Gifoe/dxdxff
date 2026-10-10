"""Read actual frozen caches, audit waveform validity without loading export labels.

The trusted historical pickle includes clinical fields; only the explicitly
allowlisted identity/waveform/timestamp fields below are used. Outer labels are
never read from the feature export or used in this audit.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\a0_sss_mil_seed42_runtime')
RAW = Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl')
FEATURE = Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
EXPORT = Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
SPLIT = Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')


def norm(x):
    return ''.join(str(x).upper().split()).replace('-', '').replace('_', '')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, obj):
    (ROOT / 'audit' / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    (ROOT / 'audit').mkdir(parents=True, exist_ok=True)
    print('RAW_AUDIT_START', flush=True)
    assert sha(EXPORT) == '1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
    assert sha(SPLIT) == 'fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278'
    with np.load(EXPORT, allow_pickle=True) as d:
        patient = d['patient'].astype(str)
        channel = np.array([norm(x) for x in d['channel']])
        center = d['center'].astype(str)
    ids = set(patient)
    pairs = set(zip(patient, channel))
    assert len(ids) == 80 and len(pairs) == len(patient) == 7635
    centers = {p: str(center[np.flatnonzero(patient == p)[0]]) for p in ids}
    with FEATURE.open('rb') as f:
        payload = pickle.load(f)
    expected = []
    for r in payload['run_records']:
        p = str(r['subject_id'])
        if p not in ids:
            continue
        s = r['sample']
        expected.append((p, str(r['run_id']), str(s.get('sample_id', r['run_id'])),
                         tuple(norm(n) for n in r['channel_names_norm']),
                         np.asarray(s['window_relative_centers_sec'], dtype=float)))
    del payload
    gc.collect()
    assert len(expected) == 256
    print('FEATURE_METADATA_MATCHED_256', flush=True)
    raw_sha = sha(RAW)
    feature_sha = sha(FEATURE)
    print('CACHE_HASHES_COMPLETE', flush=True)
    with RAW.open('rb') as f:
        payload = pickle.load(f)
    bykey = defaultdict(list)
    for r in payload['run_records']:
        if str(r['subject_id']) in ids:
            bykey[(str(r['subject_id']), str(r['run_id']))].append(r)
    counts = Counter()
    group = defaultdict(Counter)
    available = set()
    union = set()
    private = []
    for p, run, sampleid, names, times in expected:
        rr = [r for r in bykey[(p, run)]
              if str(r['sample'].get('sample_id', r['run_id'])) == sampleid
              and tuple(norm(n) for n in r['channel_names_norm']) == names]
        assert len(rr) == 1, 'raw matching not unique'
        r = rr[0]
        s = r['sample']
        wave = np.asarray(s['raw_waveform'])
        fs = float(s['raw_temporal_sfreq'])
        duration = float(s['raw_temporal_duration_sec'])
        start, length = int(s['raw_valid_start_sample']), int(s['raw_valid_samples'])
        assert wave.ndim == 2 and wave.shape[0] == len(names) == len(set(names))
        assert np.isfinite(fs) and fs > 0
        assert np.isclose(wave.shape[1] / fs, duration, atol=1 / fs)
        assert start >= 0 and length >= 0 and start + length <= wave.shape[1]
        assert np.allclose(times, np.asarray(s['window_relative_centers_sec']), atol=1e-6)
        assert np.isfinite(times).all() and len(np.unique(times)) == len(times)
        starts = np.rint((times + duration / 2 - 1) * fs).astype(int)
        size = int(round(2 * fs))
        boundary = (starts >= start) & (starts + size <= start + length)
        valid = np.zeros((len(names), len(times)), dtype=bool)
        for j, a in enumerate(starts):
            if boundary[j]:
                w = wave[:, a:a+size]
                valid[:, j] = np.isfinite(w).all(axis=1) & (np.ptp(w, axis=1) > 0)
        # No padding, nonfinite, or perfectly constant interval enters MIL.
        for c, v in zip(names, valid):
            union.add((p, c))
            if v.any():
                available.add((p, c))
        g = group[centers[p]]
        for key, val in {'records': 1, 'seizure_channel_incidences': len(names),
                         'valid_seizure_channel_incidences': int(valid.any(axis=1).sum()),
                         'candidate_windows': int(valid.size), 'valid_windows': int(valid.sum()),
                         'boundary_invalid_windows': int((~boundary).sum() * len(names)),
                         'invalid_records': int(not valid.any()),
                         'padded_records': int(length < wave.shape[1])}.items():
            counts[key] += val
            g[key] += val
        private.append({'patient': p, 'run': run, 'sample': sampleid, 'names': names,
                        'times': times, 'starts': starts, 'valid': valid,
                        'fs': fs, 'duration': duration, 'valid_start': start, 'valid_samples': length})
    assert union == pairs, 'raw/engineered canonical population mismatch'
    assert counts['seizure_channel_incidences'] == 24995
    # Preserve every patient; fallback policy is audited rather than deleting it.
    fallback = pairs - available
    no_patient_evidence = [p for p in ids if not any(q == p for q, c in available)]
    gate = not no_patient_evidence
    report = {'status': 'PASS' if gate else 'EXPERIMENT_BLOCKED', 'matched_patients': len(ids),
              'matched_records': 256, 'canonical_channels': len(pairs), **dict(counts),
              'valid_raw_canonical_channels': len(available), 'fallback_channels': len(fallback),
              'patients_without_valid_raw': len(no_patient_evidence),
              'independently_verified_edf_onsets': 0, 'clinical_onset_verified': False,
              'time_reference': 'cache-relative; nominal duration midpoint is not independently verified onset',
              'raw_sha256': raw_sha, 'feature_cache_sha256': feature_sha,
              'export_sha256': sha(EXPORT), 'split_sha256': sha(SPLIT),
              'clinical_labels_used': False, 'outer_test_labels_read': False,
              'label_fields_in_trusted_pickle_ignored': True}
    dump('RAW_ALIGNMENT_AUDIT.json', report)
    dump('RAW_FEATURE_ALIGNMENT_AUDIT.json', {'status': 'PASS', 'exact_sample_and_channel_order': True,
                                             'exact_canonical_union': True, 'records': 256,
                                             'channels': 7635, 'duplicate_windows': 0})
    for name in ['RAW_COVERAGE_BY_CENTER.csv', 'RAW_VALIDITY_AUDIT.csv']:
        rows = []
        for c in sorted(group):
            rows.append({'center': c, 'patients': sum(centers[p] == c for p in ids),
                         'canonical_channels': sum(centers[p] == c for p, ch in pairs),
                         'valid_raw_channels': sum(centers[p] == c for p, ch in available),
                         'fallback_channels': sum(centers[p] == c for p, ch in fallback), **group[c]})
        with (ROOT / 'audit' / name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # Private ledger contains no clinical labels or outcomes.
    with (ROOT / 'RAW_LEDGER_PRIVATE.pkl').open('wb') as f:
        pickle.dump({'records': private, 'raw_hash': raw_sha, 'feature_hash': feature_sha}, f)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
