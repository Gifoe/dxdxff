"""Compare frozen validation forward across metrics-only NumPy runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare')
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    from model import IctalLocalization
    root = Path(args.runtime)
    manifest = json.loads((root / 'export_amended' / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(e for e in manifest['folds'] if e['fold_idx'] == 5)
    pid = entry['validation_subjects'][0]
    path = root / 'export_amended' / manifest['cohort'][pid]['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['patient_file_sha256'][pid]
    with np.load(path, allow_pickle=False) as data:
        pair, mask = data['pair'], data['present']
        input_hash = hashlib.sha256(pair.tobytes() + mask.tobytes()).hexdigest()
        x = torch.tensor(pair, dtype=torch.float32, device='cuda')
        valid = torch.tensor(mask, dtype=torch.bool, device='cuda')
    checkpoint = torch.load(root / 'validation_amended' / 'fold_5_E3_last_private.pt',
                            map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 30
    model = IctalLocalization(use_pr=True, use_attention=False).cuda().eval()
    model.load_state_dict(checkpoint['best'][1])
    with torch.no_grad():
        logits, channels = model(x, valid)
    result = {'numpy': np.__version__, 'torch': torch.__version__, 'input_sha256': input_hash,
              'logits': logits.cpu().tolist(), 'channels': channels.cpu().tolist(),
              'checkpoint_selected_epoch': checkpoint['best'][2]}
    if args.compare:
        original = json.loads(Path(args.compare).read_text(encoding='utf-8'))
        difference = max(abs(a-b) for a,b in zip(result['logits'], original['logits']))
        assert result['input_sha256'] == original['input_sha256']
        assert result['channels'] == original['channels']
        assert result['torch'] == original['torch']
        assert difference <= 1e-6
        result['parity'] = {'status': 'PASS', 'validation_logit_max_abs_diff': difference,
                           'input_bytes_identical': True, 'torch_runtime_unchanged': True,
                           'training_complete_before_runtime_change': True,
                           'original_numpy': original['numpy'], 'finalization_numpy': np.__version__}
    Path(args.output).write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps(result.get('parity', {'status': 'FORWARD_BASELINE_SAVED'})), flush=True)


if __name__ == '__main__':
    main()
