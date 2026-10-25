"""Release-only checks of anonymous payload bytes and historical aggregate reuse.

Run from either a checkout or a real Git-tree archive. No private data imports.
This does not replace the server's independent patient-endpoint verification.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / 'experiments/drechan_loco_five_models_8seed_v1'
PUB = HERE / 'public'
OLD = REPO / 'experiments/drechan_loco_five_models_3seed_v1/public'
METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy',
           'mrr', 'hit_at_1', 'ez_fraction_bias']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check_manifest(folder, count):
    manifest = read(folder / 'PACKAGE_MANIFEST.json')
    assert manifest['private_artifacts_exported'] is False
    assert len(manifest['files']) == count
    for name, expected in manifest['files'].items():
        target = (folder / name).resolve()
        assert folder.resolve() in target.parents, name
        assert sha(target) == expected, ('PUBLIC_PAYLOAD_CHANGED', name)
    for name, expected in manifest['source'].items():
        assert sha(HERE / name) == expected, ('FROZEN_SOURCE_CHANGED', name)
    assert len(manifest['source']) == 12
    return manifest


def main():
    # Required parent math and executable metric dependencies must be in the tree.
    for name in ['common.py', 'pipeline.py', 'svm.py', 'timeconv_engine.py']:
        assert (OLD.parent / name).is_file()
        assert (REPO / 'experiments/drechan_loco_five_models_seed42_v1' / name).is_file()
    assert (REPO / 'experiments/m8_raw_backbones_pr_seed42_v1/evaluate.py').is_file()
    final = check_manifest(PUB, 42)
    admission = check_manifest(HERE / 'public_admission', 21)
    assert final['schema'] == 'LOCO5_8SEED_V1'
    assert admission['schema'] == 'LOCO5_8SEED_ADMISSION_V1'
    assert final['source'] == admission['source']
    for seed in range(45, 50):
        for name in ['ADMISSION.json', 'ROLE_AUDIT.csv', 'INPUT_AUDIT.json',
                     'INITIALIZATION_AUDIT.csv']:
            assert (PUB / f'admission_seed{seed}' / name).read_bytes() == (
                HERE / 'public_admission' / f'seed{seed}' / name).read_bytes()

    # All original 42-44 aggregate endpoints remain unchanged, including sources.
    historical_rows = 0
    for filename, key in [('SEED_RESULTS.csv', ('train_seed', 'arm')),
                          ('SOURCE_SEED_RESULTS.csv', ('train_seed', 'arm', 'held_source'))]:
        current = {tuple(r[k] for k in key): r for r in rows(PUB / filename)}
        for old in rows(OLD / filename):
            new = current[tuple(old[k] for k in key)]
            assert all(abs(float(new[m]) - float(old[m])) < 1e-12 for m in METRICS)
            historical_rows += 1
    assert historical_rows == 75

    current_cells = {(r['train_seed'], r['arm'], r['held_index']): r
                     for r in rows(PUB / 'TRAIN_CELLS.csv')}
    old_cells = rows(OLD / 'TRAIN_CELLS.csv')
    assert len(current_cells) == 160 and len(old_cells) == 60
    for old in old_cells:
        new = current_cells[(old['train_seed'], old['arm'], old['held_index'])]
        for field in ['initial_sha256', 'checkpoint_sha256', 'preprocessor_sha256',
                      'validation_sha256', 'rule']:
            assert new[field] == old[field], ('HISTORICAL_CELL_CHANGED', field)
        for field in ['epoch', 'epochs', 'threshold', 'parameters']:
            # SVM parameter count is identically serialized as "nan" in both
            # historical and new tables; exact bytes are the correct gate.
            if new[field] == old[field]:
                continue
            if old[field]:
                assert abs(float(new[field]) - float(old[field])) < 1e-12
            else:
                assert new[field] == old[field]
        assert new['reused'] == 'True'

    groups = defaultdict(lambda: [0, 0, 0])
    raw_arms = {'OURS_RAW_FEATURE', 'OURS_RAW', 'TIMECONVCNN_RAW'}
    for r in rows(PUB / 'COVERAGE.csv'):
        assert r['raw_consumed'] == str(r['arm'] in raw_arms)
        total = groups[(r['train_seed'], r['arm'])]
        for i, name in enumerate(['patients', 'channels', 'valid_raw_windows']):
            total[i] += int(r[name])
    assert len(groups) == 40
    for (_, arm), value in groups.items():
        # Feature-only arms intentionally consume no raw waveform windows.
        assert value == [80, 7635, 1471965 if arm in raw_arms else 0]
    check = read(PUB / 'INDEPENDENT_VERIFICATION.json')
    assert check['status'] == 'PASS'
    assert (check['patient_rows'], check['metric_values'], check['cells'],
            check['reused'], check['fresh'], check['nominal_intervals'],
            check['family_intervals']) == (3200, 28800, 160, 60, 100, 405, 8)
    print('PUBLIC_42_FINAL_21_ADMISSION_PAYLOAD_HASHES_12_SOURCES_PASS')
    print('HISTORICAL_60_CELL_BINDINGS_75_AGGREGATE_ROWS_UNCHANGED_PASS')
    print('ALL_40_GROUPS_80_PATIENTS_7635_CHANNELS_24_RAW_GROUPS_1471965_WINDOWS_PASS')


if __name__ == '__main__':
    main()
