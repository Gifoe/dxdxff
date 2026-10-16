"""Build an allowlisted manifest only after the independent internal terminal.

No model/data import, private rows, logs, checkpoints or recursive export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


FILES = [
    'RUN_STATUS.json', 'INTERNAL_TERMINAL.json', 'INDEPENDENT_VERIFICATION.json',
    'INTERNAL_CELL_RESULTS.csv', 'RACING_DECISIONS.json',
    'FIRST_PATH_MAIN_RESULTS.csv', 'FIRST_PATH_FOLD_RESULTS.csv',
    'FIRST_PATH_SOURCE_RESULTS.csv', 'FIRST_PATH_ABSOLUTE_INTERVALS.csv',
    'FIRST_PATH_PAIRED_INTERVALS.csv', 'ADDITIONAL_PATH_RESULTS.csv',
    'INPUT_ADMISSION.json', 'GRADIENT_ADMISSION.json',
    'EXECUTION_SOURCE_ADMISSION.json', 'NORMALIZATION_INPUT_AUDIT.csv',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    read = lambda n: json.loads((root / n).read_text(encoding='utf-8'))
    terminal = read('INTERNAL_TERMINAL.json')
    verification = read('INDEPENDENT_VERIFICATION.json')
    status = read('RUN_STATUS.json')
    assert terminal['status'] == 'COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'
    assert terminal['new_Outer_passes'] == 0
    assert verification['status'] == 'PASS'
    assert status['status'] == 'ALL_REGISTERED_INTERNAL_CELLS_COMPLETE'
    decisions = read('RACING_DECISIONS.json')['decisions']
    assert len(decisions) == 5 and {r['fold'] for r in decisions} == set(range(1, 6))
    expected = 45 + 4 * sum(r['advanced'] for r in decisions)
    assert terminal['executed_cells'] == status['actual_cells'] == verification['cells'] == expected
    assert 45 <= expected <= 65 and len(verification['checks']) == expected
    records = []
    for name in FILES:
        path = root / name
        assert path.is_file() and path.stat().st_size < 2_000_000
        row_count = None
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                assert not set(reader.fieldnames or []) & {
                    'patient', 'patient_id', 'channel', 'channel_id', 'archive_indices',
                    'logit_nez', 'prediction', 'embedding', 'path', 'edf_path',
                }
                row_count = sum(1 for _ in reader)
            if name == 'FIRST_PATH_MAIN_RESULTS.csv':
                assert row_count == 9
            if name == 'FIRST_PATH_FOLD_RESULTS.csv':
                assert row_count == 45
            if name == 'INTERNAL_CELL_RESULTS.csv':
                assert row_count == expected
            if name == 'ADDITIONAL_PATH_RESULTS.csv':
                assert row_count == expected - 45
        records.append({'file': name, 'sha256': sha(path),
                        'bytes': path.stat().st_size, 'aggregate_rows': row_count})
    result = {'status': 'PASS', 'export_scope': 'explicit_anonymous_aggregate_allowlist',
              'executed_cells': expected, 'new_Outer_passes': 0,
              'input_binding_sha256': verification['input_binding_sha256'],
              'private_files_exported': 0, 'files': records}
    target = root / 'PUBLIC_EXPORT_MANIFEST.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'files': len(records), 'cells': expected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    main(parser.parse_args().results)
