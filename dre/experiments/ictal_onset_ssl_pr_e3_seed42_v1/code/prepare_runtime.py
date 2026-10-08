"""Private-server preparation; print aggregate audit evidence only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--runtime', required=True)
    a = p.parse_args()
    package = Path(a.package)
    runtime = Path(a.runtime)
    sys.path.insert(0, str(package))
    from audit_export import extract, identity, digest, names, value as field_value, frozen_folds
    feature = Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
    raw = Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl')
    folds = runtime / 'frozen_folds.json'
    start = time.perf_counter()
    expected = json.loads((package / 'expected_sha.json').read_text())
    hashes = {'feature_cache': digest(feature), 'raw_cache': digest(raw)}
    if hashes != expected:
        raise RuntimeError('FROZEN_CACHE_HASH_MISMATCH')
    subjects = {sid for f in json.loads(folds.read_text())['folds'] for sid in f['test_subjects']}
    with feature.open('rb') as f:
        feature_payload = pickle.load(f)
    feature_records = [r for r in feature_payload['run_records'] if str(r.get('subject_id')) in subjects]
    required = {identity(r) for r in feature_records}
    selected_index = {sid: feature_payload['patient_index'][sid] for sid in subjects}
    feature_by_key = {identity(r): r for r in feature_records}
    normalized_folds = frozen_folds(folds, subjects)
    n_unique_channels = sum(len(r['canonical_channels']) for r in selected_index.values())
    del feature_payload
    with raw.open('rb') as f:
        payload = pickle.load(f)
    keys = Counter()
    paths = Counter()
    center_paths = Counter()
    private_rows = []
    invalid_rows = []
    all_issues = []
    raw_incidences = source_windows = 0
    source_flags = Counter()
    alternative_padding = Counter()
    matched_by_patient = Counter()
    remaining_channel_names = {}
    raw_keys = Counter()
    for rec in payload['run_records']:
        if identity(rec) not in required:
            continue
        raw_keys[identity(rec)] += 1
        matched_by_patient[str(rec['subject_id'])] += 1
        fields = {**rec, **(rec.get('sample') or {})}
        row = {'key': identity(rec), 'paths': {}}
        sample = rec.get('sample') or {}
        import numpy as np
        sig = np.asarray(sample.get('raw_waveform'))
        feat = feature_by_key[identity(rec)]
        raw_names, feature_names = names(rec), names(feat)
        raw_incidences += len(feature_names)
        source_windows += len(feature_names) * len(field_value(feat, 'window_relative_centers_sec', []))
        issues = []
        if set(raw_names) != set(feature_names):
            issues.append('CHANNEL_MISMATCH')
        if sig.shape != (len(raw_names), 15000) or float(sample.get('raw_temporal_sfreq', 0)) != 250:
            issues.append('RAW_SHAPE_RATE_MISMATCH')
        if not np.isfinite(sig).all():
            issues.append('NONFINITE_RAW')
        if 'raw_valid_start_sample' not in sample or 'raw_valid_samples' not in sample:
            issues.append('MISSING_VALID_INTERVAL')
        valid_start = int(sample.get('raw_valid_start_sample', 0))
        valid_end = valid_start + int(sample.get('raw_valid_samples', 15000))
        if not 0 <= valid_start < valid_end <= 15000:
            issues.append('INVALID_VALID_INTERVAL')
        for window_name, begin, end in [('pre_minus10_to_0', 5000, 7500),
                                        ('pre_minus11_to_minus1', 4750, 7250)]:
            overlap = max(0, min(end, valid_end) - max(begin, valid_start))
            if overlap != end - begin:
                alternative_padding[window_name] += 1
        phase_details = {}
        for phase, begin, end in [('preictal', 3750, 6250), ('early_ictal', 7500, 10000)]:
            overlap = max(0, min(end, valid_end) - max(begin, valid_start))
            if overlap != end - begin:
                issues.append(phase.upper() + '_INCLUDES_PADDING')
                phase_details[phase] = {'required_start': begin, 'required_end': end,
                                        'padded_samples': (end - begin) - overlap}
        if phase_details:
            invalid_rows.append({'patient': str(rec['subject_id']), 'key': identity(rec),
                                 'channels': len(feature_names), 'valid_start': valid_start,
                                 'valid_end': valid_end, 'phase_details': phase_details})
        else:
            remaining_channel_names.setdefault(str(rec['subject_id']), set()).update(feature_names)
        all_issues.extend(issues)
        row['issues'] = issues
        row['valid_interval'] = [valid_start, valid_end]
        source_flags[str(sample.get('raw_temporal_onset_centered', sample.get('onset_centered', 'MISSING')))] += 1
        for name, value in fields.items():
            if any(part in name.lower() for part in ('onset', 'edf', 'path', 'file', 'source', 'start', 'end', 'time')):
                keys[name] += 1
            if isinstance(value, str) and (value.lower().endswith(('.edf', '.bdf', '.tsv', '.json')) or ':\\' in value):
                exists = Path(value).is_file()
                paths[name + ('_exists' if exists else '_missing')] += 1
                row['paths'][name] = {'path': value, 'exists': exists}
                if exists:
                    center_paths[str(rec.get('subject_id', '')).split(':')[0]] += 1
        private_rows.append(row)
    provenance = {
        'source_status': 'SOURCE_CODE_SUPPORTED',
        'independent_status': 'INDEPENDENT_EDF_UNCONFIRMED',
        'matched_records': len(private_rows),
        'independent_edf_confirmed_records': 0,
        'metadata_field_counts': dict(keys),
        'stored_source_path_counts': dict(paths),
        'centers_with_existing_path_fields': dict(center_paths),
        'source_only_scope': 'validation_only; independent onset gate required before outer testing',
        'elapsed_seconds': time.perf_counter() - start,
    }
    (runtime / 'ONSET_PROVENANCE_AUDIT.json').write_text(json.dumps(provenance, indent=2))
    (runtime / 'ONSET_PROVENANCE_PRIVATE.json').write_text(json.dumps(private_rows, indent=2))
    print('ONSET_PROVENANCE', json.dumps(provenance), flush=True)
    invalid_patients = {r['patient'] for r in invalid_rows}
    if set(raw_keys) != required:
        all_issues.extend(['MISSING_MATCHED_RAW_KEY'] * len(required - set(raw_keys)))
    all_issues.extend(['DUPLICATE_MATCHED_RAW_KEY'] * sum(n - 1 for n in raw_keys.values()))
    if (len(subjects), len(private_rows), n_unique_channels, raw_incidences, source_windows) != (80, 256, 7635, 24995, 1471965):
        all_issues.append('FROZEN_COHORT_FOOTPRINT_MISMATCH')
    if len(feature_records) != len(required):
        all_issues.append('DUPLICATE_FEATURE_KEY')
    affected_by_fold = []
    for f in normalized_folds:
        entry = {'fold': f['fold_idx']}
        for role in ('fit_subjects', 'validation_subjects', 'test_subjects'):
            member = set(f[role])
            entry[role + '_affected_patients'] = len(member & invalid_patients)
            entry[role + '_affected_records'] = sum(r['patient'] in member for r in invalid_rows)
        affected_by_fold.append(entry)
    invalid_by_center = Counter(r['patient'].split(':')[0] for r in invalid_rows)
    report = {
        'status': ('BLOCKED_RAW_WINDOW_PADDING' if invalid_rows else
                   'BLOCKED_OTHER_DATA_CONTRACT' if all_issues else 'EXHAUSTIVE_PREFLIGHT_PASS'),
        'hashes': hashes, 'expected_hashes_match': True,
        'n_patients': len(subjects), 'n_seizures': len(private_rows),
        'n_unique_channels': n_unique_channels, 'run_channel_incidences': raw_incidences,
        'source_window_incidences': source_windows,
        'duplicate_feature_keys': len(feature_records) - len(required),
        'matched_raw_records': len(private_rows),
        'source_onset_flag_counts': dict(source_flags),
        'issues': dict(Counter(all_issues)),
        'affected_records': len(invalid_rows), 'affected_patients': len(invalid_patients),
        'affected_run_channel_pairs': sum(r['channels'] for r in invalid_rows),
        'unaffected_records': len(private_rows) - len(invalid_rows),
        'affected_records_by_center': dict(invalid_by_center),
        'affected_by_fold': affected_by_fold,
        'preictal_padded_samples_per_channel': sum(r['phase_details'].get('preictal', {}).get('padded_samples', 0) for r in invalid_rows),
        'early_ictal_padded_samples_per_channel': sum(r['phase_details'].get('early_ictal', {}).get('padded_samples', 0) for r in invalid_rows),
        'training_started': False, 'test_evaluation_started': False,
        'independent_onset_confirmed_records': 0,
        'alternative_protocol_diagnostics_not_authorized_for_training': {
            'pre_minus10_to_0_padding_records': alternative_padding['pre_minus10_to_0'],
            'pre_minus11_to_minus1_padding_records': alternative_padding['pre_minus11_to_minus1'],
            'exclude_invalid_records_remaining_records': len(private_rows) - len(invalid_rows),
            'exclude_invalid_records_patients_retained': len(remaining_channel_names),
            'exclude_invalid_records_unique_channels_retained': sum(map(len, remaining_channel_names.values())),
            'note': 'Read-only feasibility checks. No window, cohort, cache, or label was changed.',
        },
        'required_action': 'Resolve source completeness or explicitly authorize a new window/cohort protocol; original E3 cannot train with invalid crops.',
    }
    (runtime / 'DATA_BLOCKER_AUDIT.json').write_text(json.dumps(report, indent=2))
    (runtime / 'PADDING_RECORDS_PRIVATE.json').write_text(json.dumps(invalid_rows, indent=2))
    print('EXHAUSTIVE_PREFLIGHT', json.dumps(report), flush=True)
    if all_issues:
        (runtime / 'RUN_STATUS.json').write_text(json.dumps({'status': report['status'],
            'data_audit': 'DATA_BLOCKER_AUDIT.json', 'training_started': False}, indent=2))
        return
    result = extract(feature, raw, folds, runtime / 'export',
                     allow_source_only=True, expected_sha=package / 'expected_sha.json')
    print('CACHE_AUDIT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
