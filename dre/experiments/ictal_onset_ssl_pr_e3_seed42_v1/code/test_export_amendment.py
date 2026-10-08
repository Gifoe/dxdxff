import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_export as a


def fixture(tmp_path, monkeypatch):
    feature = {'patient_index': {}, 'run_records': []}
    raw = {'run_records': []}
    for pid, count, runs in [('fixtureA', 104, 2), ('fixtureB', 2, 1)]:
        names = [f'LA{i + 1}' for i in range(count)]
        labels = [i % 2 for i in range(count)]
        feature['patient_index'][pid] = {'canonical_channels': names, 'labels': labels, 'center': 'fixture'}
        for run in range(runs):
            sample = {'sample_id': f'run{run}', 'source_seizure_id': f'seizure{run}',
                      'seizure_onset_sec': 11. if run == 0 and pid == 'fixtureA' else 30.,
                      'start_sec': 0., 'end_sec': 60., 'channel_names_norm': names,
                      'window_relative_centers_sec': np.array([-14., -10., 0., 4.])}
            rec = {'subject_id': pid, 'run_id': f'run{run}', 'sample': sample}
            feature['run_records'].append(rec)
            raw_sample = dict(sample)
            bad = run == 0 and pid == 'fixtureA'
            raw_sample.update(raw_waveform=np.ones((count, 15000), dtype='float32'),
                              raw_temporal_sfreq=250., raw_temporal_duration_sec=60.,
                              raw_valid_start_sample=4750 if bad else 0,
                              raw_valid_samples=10250 if bad else 15000)
            raw['run_records'].append({**rec, 'sample': raw_sample})
    for name, obj in [('feature', feature), ('raw', raw)]:
        with (tmp_path / f'{name}.pkl').open('wb') as f:
            pickle.dump(obj, f)
    (tmp_path / 'folds.json').write_text(json.dumps({'folds': [{'test_subjects': ['fixtureA', 'fixtureB']}]}))
    excluded = [{'key': a.identity(raw['run_records'][0])}]
    (tmp_path / 'excluded.json').write_text(json.dumps(excluded))
    (tmp_path / 'amendment.json').write_text(json.dumps({'approval': 'USER_APPROVED',
        'authorized_exclusion_records': 1, 'authorized_exclusion_channels': 104}))
    monkeypatch.setattr(a, 'EXPECTED_PATIENTS', 2)
    monkeypatch.setattr(a, 'EXPECTED_SEIZURES', 3)
    monkeypatch.setattr(a, 'EXPECTED_CHANNELS', 106)
    monkeypatch.setattr(a, 'frozen_folds', lambda path, patients: [])
    return tmp_path / 'feature.pkl', tmp_path / 'raw.pkl', tmp_path / 'folds.json'


def test_authorized_exclusion_preserves_all_patients_and_channels(tmp_path, monkeypatch):
    feature, raw, folds = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='include padding'):
        a.extract(feature, raw, folds, tmp_path / 'blocked', allow_source_only=True)
    with pytest.raises(ValueError, match='explicit protocol amendment'):
        a.extract(feature, raw, folds, tmp_path / 'blocked2', allow_source_only=True,
                  approved_exclusion_private=tmp_path / 'excluded.json')
    audit = a.extract(feature, raw, folds, tmp_path / 'approved', allow_source_only=True,
        approved_exclusion_private=tmp_path / 'excluded.json', amendment_json=tmp_path / 'amendment.json')
    assert audit['n_seizures'] == 2 and audit['source_n_seizures'] == 3
    assert audit['n_patients'] == 2 and audit['n_unique_channels'] == 106
    assert audit['authorized_excluded_records'] == 1
    assert audit['authorized_excluded_run_channel_pairs'] == 104
    assert audit['valid_pre_ictal_pairs'] == 106
    manifest = json.loads((tmp_path / 'approved' / 'manifest.json').read_text())
    assert len(manifest['patient_file_sha256']) == 2


def test_exclusion_of_valid_record_is_rejected(tmp_path, monkeypatch):
    feature, raw, folds = fixture(tmp_path, monkeypatch)
    with feature.open('rb') as f:
        obj = pickle.load(f)
    (tmp_path / 'excluded.json').write_text(json.dumps([{'key': a.identity(obj['run_records'][1])}]))
    with pytest.raises(ValueError, match='include padding'):
        a.extract(feature, raw, folds, tmp_path / 'wrong', allow_source_only=True,
            approved_exclusion_private=tmp_path / 'excluded.json', amendment_json=tmp_path / 'amendment.json')
