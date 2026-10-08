"""Read-only verification of E3 artifacts; emits aggregate audit only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--original-model', required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    out = runtime / 'validation_amended'
    status = read(out / 'RUN_STATUS.json')
    assert status['status'] == 'VALIDATION_COMPLETE_TEST_NOT_ACCESSED'
    assert status['outer_test_accessed'] is False
    binding = read(out / 'TRAINING_PROTOCOL_LOCK.json')
    assert binding['model'] == sha(args.original_model)
    assert binding['manifest'] == sha(runtime / 'export_amended' / 'manifest.json')
    assert binding['audit'] == sha(runtime / 'export_amended' / 'audit.json')
    manifest = read(runtime / 'export_amended' / 'manifest.json')
    selections = read(out / 'validation_selections.json')
    assert len(selections) == 5
    expected_val = set()
    epochs = []
    for fold in manifest['folds']:
        fit, val, test = map(set, (fold['fit_subjects'], fold['validation_subjects'], fold['test_subjects']))
        assert not (fit & val or fit & test or val & test)
        expected_val.update(val)
    for entry in selections:
        fold = entry['fold']
        for stage, count in [('ssl', 25), ('E3', 30)]:
            saved = torch.load(out / f'fold_{fold}_{stage}_last_private.pt', map_location='cpu', weights_only=False)
            assert saved['binding'] == binding and saved['epoch'] == count
            assert len(saved['history']) == count
        path = Path(entry['checkpoint'])
        assert sha(path) == entry['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['binding'] == binding
        assert checkpoint['selected_epoch'] == entry['selected_epoch']
        assert checkpoint['ez_threshold'] == entry['threshold']
        assert checkpoint['val_metrics'] == entry['val_metrics']
        assert all(np.isfinite(list(entry['val_metrics'].values())))
        epochs.append({'fold': fold, 'ssl_epochs': 25, 'supervised_epochs': 30,
                       'selected_epoch': entry['selected_epoch'], 'threshold': entry['threshold'],
                       'checkpoint_sha256': entry['checkpoint_sha256']})
    with (out / 'validation_per_patient_private.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 65
    assert {r['patient'] for r in rows} == expected_val
    summary = read(out / 'validation_summary.json')
    assert summary['n_unique_validation_patients'] == len(expected_val)
    maximum_error = 0.
    for metric, value in summary['fold_mean'].items():
        maximum_error = max(maximum_error, abs(value - np.mean([s['val_metrics'][metric] for s in selections])))
    for metric, value in summary['unique_patient_mean'].items():
        independent = np.mean([np.mean([float(r[metric]) for r in rows if r['patient'] == pid])
                               for pid in expected_val])
        maximum_error = max(maximum_error, abs(value - independent))
    assert maximum_error < 1e-12
    with (out / 'ssl_history.csv').open(encoding='utf-8', newline='') as handle:
        ssl = list(csv.DictReader(handle))
    with (out / 'supervised_history.csv').open(encoding='utf-8', newline='') as handle:
        supervised = list(csv.DictReader(handle))
    assert len(ssl) == 125 and len(supervised) == 150
    report = {'status': 'PASS', 'original_model_sha256_match': True,
              'protocol_bindings_match': True, 'selected_checkpoint_hashes_match': True,
              'fit_val_test_disjoint_in_every_fold': True,
              'n_validation_cells': len(rows), 'n_unique_validation_patients': len(expected_val),
              'ssl_epochs_completed': 125, 'supervised_epochs_completed': 150,
              'summary_recalculation_max_abs_diff': maximum_error,
              'outer_test_accessed': False, 'folds': epochs,
              'scope': 'Artifact consistency and source-only validation; no model inference or independent test'}
    (out / 'COMPLETION_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
