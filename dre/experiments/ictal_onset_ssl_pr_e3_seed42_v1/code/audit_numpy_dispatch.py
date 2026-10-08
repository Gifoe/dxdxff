"""Synthetic equivalence gate for NumPy execution alternatives; no real predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare')
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    from run_experiment import metric_one, fit_threshold
    rng = np.random.RandomState(42)
    values, thresholds, rows = [], [], []
    sampling = hashlib.sha256()
    for index in range(60):
        count = 8 + index * 3
        labels = rng.randint(0, 2, count)
        labels[:2] = [0, 1]
        scores = rng.uniform(0, 1, count).astype(np.float32)
        sampling.update(rng.permutation(count).tobytes())
        rows.append({'labels_ez': labels, 'scores_ez': scores})
        for threshold in (0.05, .3, .5, .7, .95):
            metrics = metric_one(labels, scores, threshold)
            values.extend(metrics[key] for key in sorted(metrics))
        if len(rows) == 5:
            thresholds.append(fit_threshold(rows))
            rows = []
    result = {'disabled_cpu_features': os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),
              'metrics': values, 'thresholds': thresholds, 'sampling_sha256': sampling.hexdigest()}
    if args.compare:
        original = json.loads(Path(args.compare).read_text(encoding='utf-8'))
        difference = float(np.max(np.abs(np.asarray(values) - original['metrics'])))
        assert difference == 0.
        assert thresholds == original['thresholds']
        assert result['sampling_sha256'] == original['sampling_sha256']
        result['parity'] = {'status': 'PASS', 'metric_max_abs_diff': difference,
                            'thresholds_exact': True, 'sampling_exact': True,
                            'synthetic_patients': 60, 'synthetic_metric_points': 300,
                            'scope': 'Synthetic metric/threshold/sampling parity; no real predictions'}
    Path(args.output).write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps(result.get('parity', {'status': 'BASELINE_SAVED'})), flush=True)


if __name__ == '__main__':
    main()
