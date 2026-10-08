"""E1 original-loop and exact resume tests; synthetic patients only."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    from model import IctalLocalization
    from run_experiment import PatientFiles, train_one, set_seed
    from run_e3_validation import CachedPatientFiles
    from run_e1_validation import train_e1
    torch.set_num_threads(2)
    set_seed(142)
    e1 = IctalLocalization(use_pr=True, use_attention=False)
    set_seed(142)
    e3 = IctalLocalization(use_pr=True, use_attention=False)
    result = {'same_supervised_initialization': all(torch.equal(v, e3.state_dict()[k])
              for k, v in e1.state_dict().items()),
              'parameters': sum(p.numel() for p in e1.parameters()),
              'same_architecture_as_E3': str(e1) == str(e3), 'device': 'cpu'}
    class Interrupted(RuntimeError):
        pass
    def stop(*args):
        if args[1] == 1:
            raise Interrupted('durable complete-epoch interruption')
    def noop(*args):
        pass
    with tempfile.TemporaryDirectory(prefix='e1_parity_') as temporary:
        root = Path(temporary)
        (root / 'patients').mkdir()
        original, resumed = root / 'original', root / 'resumed'
        original.mkdir()
        resumed.mkdir()
        rng, cohort = np.random.default_rng(42), {}
        for i in range(4):
            pair = rng.normal(size=(1, 4, 2, 2500)).astype('float32')
            pair[:, :2, 1] *= 2
            np.savez_compressed(root / 'patients' / f'{i}.npz', pair=pair,
                                present=np.ones((1, 4), dtype=bool),
                                labels_ez=np.array([1, 1, 0, 0], dtype='float32'))
            cohort[str(i)] = {'file': f'patients/{i}.npz', 'center': 'fixture'}
        manifest = {'cohort': cohort}
        fit, val = ['0', '1'], ['2']
        loader = PatientFiles(root, manifest, 'cpu')
        cached = CachedPatientFiles(root, manifest, 'cpu', fit + val, verify=False)
        result['input_max_abs_diff'] = max(float((loader.get(p)[0] - cached.get(p)[0]).abs().max())
                                           for p in fit + val)
        try:
            cached.get('3')
        except RuntimeError as exc:
            assert str(exc) == 'OUTER_TEST_ACCESS_FORBIDDEN'
            result['outer_access_rejected'] = True
        else:
            raise AssertionError('outer patient access accepted')
        baseline, _ = train_one(1, 'E1', loader, fit, val, original, 2, 142)
        try:
            train_e1(1, cached, fit, val, resumed, 2, 142, {'fixture': 1}, stop)
        except Interrupted:
            pass
        else:
            raise AssertionError('interruption not exercised')
        selected, _, _ = train_e1(1, cached, fit, val, resumed, 2, 142, {'fixture': 1}, noop)
        original_weights = torch.load(baseline['checkpoint'], weights_only=True)['state_dict']
        resumed_weights = torch.load(selected['checkpoint'], weights_only=True)['state_dict']
        result['original_vs_resumed_max_parameter_diff'] = max(float((v - resumed_weights[k]).abs().max())
                                                               for k, v in original_weights.items())
        result['selected_epoch_exact'] = baseline['selected_epoch'] == selected['selected_epoch']
        result['threshold_exact'] = baseline['threshold'] == selected['threshold']
        result['metrics_exact'] = baseline['val_metrics'] == selected['val_metrics']
        result['no_ssl_artifacts'] = not any('ssl' in p.name.lower() for p in resumed.iterdir())
        try:
            train_e1(1, cached, fit, val, resumed, 2, 142, {'fixture': 2}, noop)
        except RuntimeError as exc:
            assert str(exc) == 'RESUME_PROTOCOL_OR_SOURCE_CHANGED'
            result['changed_binding_rejected'] = True
        else:
            raise AssertionError('changed binding accepted')
    assert result['parameters'] == 44401
    assert result['input_max_abs_diff'] == result['original_vs_resumed_max_parameter_diff'] == 0.
    assert all(result[key] for key in ('same_supervised_initialization', 'same_architecture_as_E3',
        'outer_access_rejected', 'selected_epoch_exact', 'threshold_exact', 'metrics_exact',
        'no_ssl_artifacts', 'changed_binding_rejected'))
    result['status'] = 'PASS'
    result['scope'] = 'Synthetic E1 original-loop/resume parity; not scientific performance'
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
