"""Read-only E1 completion and paired-summary verification; no model inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    for name in ('runtime', 'e3-runtime', 'protocol'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, e3 = Path(args.runtime), Path(args.e3_runtime)
    out, old = root / 'validation', e3 / 'validation_amended'
    status, binding = read(out / 'RUN_STATUS.json'), read(out / 'TRAINING_BINDING.json')
    assert status['status'] == 'E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED'
    assert status['outer_test_accessed'] is False
    assert sha(args.protocol) == binding['protocol']
    assert sha(old / 'score_selection_lock.json') == binding['e3_selection_lock']
    assert sha(old / 'validation_per_patient_private.csv') == binding['e3_private_validation_ledger']
    manifest = read(e3 / 'export_amended' / 'manifest.json')
    assert sha(e3 / 'export_amended' / 'manifest.json') == binding['export_manifest_sha256']
    selected = read(out / 'validation_selections.json')
    assert len(selected) == 5
    for selection in selected:
        assert selection['variant'] == 'E1' and not selection['ssl_checkpoint_loaded']
        assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
        last = torch.load(out / f"fold_{selection['fold']}_E1_last_private.pt", map_location='cpu', weights_only=False)
        assert last['binding'] == binding and last['epoch'] == 30
        assert len(last['history']) == 30
        final = torch.load(selection['checkpoint'], map_location='cpu', weights_only=True)
        assert final['binding'] == binding
        assert final['val_metrics'] == selection['val_metrics']
        assert final['selected_epoch'] == selection['selected_epoch']
        assert final['ez_threshold'] == selection['threshold']
    assert len(rows(out / 'supervised_history.csv')) == 150
    assert not any('ssl' in path.name.lower() for path in out.glob('*.pt'))
    for selection in read(old / 'validation_selections.json'):
        assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
    e1_rows, e3_rows = rows(out / 'validation_per_patient_private.csv'), rows(old / 'validation_per_patient_private.csv')
    e1_map = {(r['patient'], int(r['fold'])): r for r in e1_rows}
    e3_map = {(r['patient'], int(r['fold'])): r for r in e3_rows}
    expected = {(pid, f['fold_idx']) for f in manifest['folds'] for pid in f['validation_subjects']}
    assert len(e1_rows) == len(e1_map) == len(expected) == 65
    assert set(e1_map) == set(e3_map) == expected
    assert all(int(e1_map[k][m]) == int(e3_map[k][m]) for k in expected for m in ('n_channels', 'ez_count'))
    ids = sorted({key[0] for key in expected})
    assert len(ids) == 47
    summary = read(out / 'validation_summary.json')
    maximum = 0.
    for metric, mean in summary['fold_mean'].items():
        independent = np.mean([float(r[metric]) for r in e1_rows])
        maximum = max(maximum, abs(mean - independent))
    paired = {r['metric']: r for r in rows(out / 'paired_bootstrap_E3_minus_E1.csv')}
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    multiplicities = np.zeros((10000, len(ids)), dtype=int)
    for i, draw in enumerate(draws):
        multiplicities[i] = np.bincount(draw, minlength=len(ids))
    counts = np.array([sum(key[0] == pid for key in expected) for pid in ids])
    for metric, reported in paired.items():
        contributions = np.array([sum(float(e3_map[k][metric]) - float(e1_map[k][metric])
                                      for k in expected if k[0] == pid) for pid in ids])
        independent = (multiplicities @ contributions) / (multiplicities @ counts)
        lower, upper = np.quantile(independent, [.025, .975])
        maximum = max(maximum, abs(float(reported['delta_patient_fold_cell_mean']) - contributions.sum() / 65),
                      abs(float(reported['cluster_bootstrap_95_lower']) - lower),
                      abs(float(reported['cluster_bootstrap_95_upper']) - upper))
    assert maximum < 1e-12
    e3_selected = read(old / 'validation_selections.json')
    f1_deltas = [b['val_metrics']['macro_f1'] - a['val_metrics']['macro_f1'] for a,b in zip(selected, e3_selected)]
    ap_delta = np.mean([b['val_metrics']['ez_ap'] - a['val_metrics']['ez_ap'] for a,b in zip(selected, e3_selected)])
    expected_gate = bool(np.mean(f1_deltas) >= .02 and ap_delta >= 0 and sum(d > 0 for d in f1_deltas) >= 4)
    assert read(out / 'validation_gate.json')['passed'] == expected_gate
    report = {'status': 'PASS', 'variant': 'E1', 'folds': 5, 'supervised_epochs_completed': 150,
              'ssl_epochs_completed': 0, 'parameters': 44401, 'pretrained_ssl_loading': False,
              'source_bindings_match': True, 'e3_selected_checkpoint_hashes_unchanged': True,
              'e3_selection_lock_and_private_ledger_unchanged': True,
              'selected_e1_checkpoint_hashes_match': True, 'matched_classification_units': True,
              'n_validation_cells': 65, 'n_unique_validation_patients': 47,
              'summary_and_paired_bootstrap_recalculation_max_abs_diff': maximum,
              'outer_test_accessed': False, 'optional_controls_run': False,
              'e3_development_gate_passed': expected_gate,
              'scope': 'Read-only artifact verification; no model inference or outer evaluation'}
    (out / 'COMPLETION_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
