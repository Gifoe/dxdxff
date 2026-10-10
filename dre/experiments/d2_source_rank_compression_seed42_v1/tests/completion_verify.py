"""Post-execution integrity only; never executes a trainer or selects an outcome."""
import argparse
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
import pandas as pd
import torch
from common import sha, write_json
from prepare import ROOT, PRIOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    pub = ROOT / 'public'
    status = json.loads((pub / 'RUN_STATUS.json').read_text())
    assert status['status'] == 'COMPLETE' and status['formal_new_runs'] == 15
    assert not status['R4_retrained'] and not status['outer_TEST_accessed']
    audited = json.loads((pub / 'ENGINEERING_TESTS.json').read_text())
    assert audited['status'] == 'PASS'
    assert audited['protocol_sha256'] == sha(args.protocol)
    code = args.protocol.parent / 'code'
    assert {p.name: sha(p) for p in code.glob('*.py')} == audited['code_sha256']
    archive = json.loads((args.protocol.parent / 'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    assert len(archive) == 101
    assert all(sha(PRIOR / name) == digest for name, digest in archive.items())
    seals = []
    for fold in range(1, 6):
        shared = []
        for arm in ['R0', 'R1', 'R2']:
            cell = ROOT / f'fold{fold}' / arm
            seal = json.loads((cell / 'FINAL_COMPLETE.json').read_text())
            assert seal['status'] == 'PASS'
            assert seal['binding']['protocol'] == sha(args.protocol)
            assert seal['binding']['code'] == audited['code_sha256']
            assert seal['best_sha256'] == sha(cell / 'BEST_PRIVATE.pt')
            assert seal['prediction_sha256'] == sha(cell / 'PREDICTIONS_PRIVATE.pt')
            shared.append(seal['binding']['shared_initial_hash'])
            seals.append({'fold': fold, 'method': arm, 'checkpoint_sha256': seal['best_sha256'],
                          'predictions_sha256': seal['prediction_sha256']})
        assert len(set(shared)) == 1
    assert status['prediction_ledger_sha256'] == sha(ROOT / 'FINAL_PREDICTION_LEDGER_PRIVATE.pt')
    summary = pd.read_csv(pub / 'VALIDATION_SUMMARY.csv')
    assert set(summary.method) == {'R0', 'R1', 'R2', 'R4'}
    assert (summary.patient_fold_appearances == 65).all() and (summary.unique_patients == 47).all()
    bootstrap = pd.read_csv(pub / 'PAIRED_BOOTSTRAP.csv')
    assert len(bootstrap) == 66 and (bootstrap.draws == 10000).all()
    assert np.isfinite(bootstrap[['delta', 'CI_low', 'CI_high']].to_numpy()).all()
    ranking = pd.read_csv(pub / 'BIAS_DIRECTION_INTERVENTION.csv')
    assert (ranking.loc[ranking.intervention == 'BIAS_OFF',
                        ['ez_auprc', 'ez_auroc', 'ez_mrr', 'top1_is_ez']] == 0).all().all()
    artifacts = {p.name: sha(p) for p in pub.iterdir() if p.is_file() and p.name != 'COMPLETION_VERIFICATION.json'}
    write_json(pub / 'COMPLETION_VERIFICATION.json', {
        'status': 'PASS', 'archived_artifacts_unchanged': 101,
        'formal_cells_verified': 15, 'code_unchanged_since_pretraining_audit': True,
        'shared_initial_state_identical_per_fold': True, 'R4_retrained': False,
        'outer_TEST_accessed': False, 'protocol_sha256': sha(args.protocol),
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'selected_cells': seals,
        'aggregate_artifacts_sha256': artifacts,
    })
    print('POSTEXECUTION_ALL_HASHES_AND_AGGREGATES_PASS', flush=True)


if __name__ == '__main__':
    main()
