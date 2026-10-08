"""Compare supplied SSL/E3 training against interrupted/resumed implementation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.package)))
    from run_experiment import PatientFiles, run_ssl, train_one
    from run_e3_validation import CachedPatientFiles, ssl_stage, supervised_stage
    torch.set_num_threads(2)
    class Interrupted(RuntimeError):
        pass
    def interrupt(stage, fold, epoch, total, row):
        if epoch == 1:
            raise Interrupted('simulated crash after durable epoch checkpoint')
    def noop(*args):
        pass
    def maximum(a, b):
        return max(float((a[k].cpu() - b[k].cpu()).abs().max()) for k in a)
    results = {}
    with tempfile.TemporaryDirectory(prefix='e3_resume_parity_') as temporary:
        root = Path(temporary)
        patients = root / 'patients'
        patients.mkdir()
        original, resumed = root / 'original', root / 'resumed'
        original.mkdir()
        resumed.mkdir()
        rng = np.random.default_rng(42)
        cohort = {}
        for i in range(4):
            pair = rng.normal(size=(1, 4, 2, 2500)).astype('float32')
            pair[:, :2, 1] *= 2
            present = np.ones((1, 4), dtype=bool)
            labels = np.array([1, 1, 0, 0], dtype='float32')
            np.savez_compressed(patients / f'{i}.npz', pair=pair, present=present, labels_ez=labels)
            cohort[str(i)] = {'file': f'patients/{i}.npz', 'center': 'fixture'}
        manifest = {'cohort': cohort}
        fit, val = ['0', '1'], ['2']
        loader = PatientFiles(root, manifest, 'cpu')
        cached = CachedPatientFiles(root, manifest, 'cpu', fit + val, verify=False)
        results['cached_input_max_abs_diff'] = max(
            float((loader.get(pid)[0] - cached.get(pid)[0]).abs().max()) for pid in fit + val)
        try:
            cached.get('3')
        except RuntimeError as exc:
            assert str(exc) == 'OUTER_TEST_ACCESS_FORBIDDEN'
            results['held_out_access_rejected'] = True
        else:
            raise AssertionError('held-out access was accepted')
        baseline_ssl, _ = run_ssl(1, loader, fit, original, 2, 43)
        try:
            ssl_stage(1, cached, fit, resumed, 2, 43, {'fixture': 1}, interrupt)
        except Interrupted:
            pass
        else:
            raise AssertionError('interruption not exercised')
        resumed_ssl, _, _ = ssl_stage(1, cached, fit, resumed, 2, 43, {'fixture': 1}, noop)
        results['ssl_original_vs_resumed_max_abs_diff'] = maximum(
            torch.load(baseline_ssl, weights_only=True), torch.load(resumed_ssl, weights_only=True))
        baseline_sel, _ = train_one(1, 'E3', loader, fit, val, original, 2, 142, baseline_ssl)
        try:
            supervised_stage(1, cached, fit, val, resumed, 2, 142, resumed_ssl, {'fixture': 1}, interrupt)
        except Interrupted:
            pass
        else:
            raise AssertionError('supervised interruption not exercised')
        resumed_sel, _, _ = supervised_stage(1, cached, fit, val, resumed, 2, 142,
                                             resumed_ssl, {'fixture': 1}, noop)
        results['supervised_original_vs_resumed_max_abs_diff'] = maximum(
            torch.load(baseline_sel['checkpoint'], weights_only=True)['state_dict'],
            torch.load(resumed_sel['checkpoint'], weights_only=True)['state_dict'])
        results['selected_epoch_parity'] = baseline_sel['selected_epoch'] == resumed_sel['selected_epoch']
        results['threshold_parity'] = baseline_sel['threshold'] == resumed_sel['threshold']
        results['metric_parity'] = baseline_sel['val_metrics'] == resumed_sel['val_metrics']
        try:
            ssl_stage(1, cached, fit, resumed, 2, 43, {'fixture': 2}, noop)
        except RuntimeError as exc:
            assert str(exc) == 'RESUME_PROTOCOL_OR_SOURCE_CHANGED'
            results['changed_resume_binding_rejected'] = True
        else:
            raise AssertionError('tampered resume protocol accepted')
    assert results['cached_input_max_abs_diff'] == 0
    assert results['ssl_original_vs_resumed_max_abs_diff'] == 0
    assert results['supervised_original_vs_resumed_max_abs_diff'] == 0
    assert all(results[k] for k in ('selected_epoch_parity', 'threshold_parity', 'metric_parity',
                                   'held_out_access_rejected', 'changed_resume_binding_rejected'))
    results['status'] = 'PASS'
    results['device'] = 'cpu'
    results['scope'] = 'synthetic original-vs-resume equivalence; no real-model outcomes'
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
