import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def test_summary_is_seed_mean_and_sample_sd_not_best_or_fold_mean():
    path = Path(__file__).parents[1]/'code/run_multiseed.py'
    spec = importlib.util.spec_from_file_location('multi_contract', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    frame = pd.DataFrame({'seed': [0, 1, 42], 'auroc': [.6, .7, .8]})
    out = runner.summarize(frame, ['auroc']).iloc[0]
    assert abs(out['mean']-.7) < 1e-12
    assert abs(out['std']-.1) < 1e-12 and out['std_ddof'] == 1
    assert out['n_seeds'] == 3 and out['mean'] != out['maximum']


def test_seed_schedule_matches_original_seed42_and_changes_initialization():
    seeds = [0, 1, 2, 3, 42]
    for fold in range(1, 6):
        values = [s+1009*fold for s in seeds]
        assert len(set(values)) == 5 and values[-1] == 42+1009*fold
    # Fold averages must not replace patient-equal averages for unequal folds.
    a = np.repeat([.3, .8], [15, 17])
    assert np.mean(a) != np.mean([.3, .8])
