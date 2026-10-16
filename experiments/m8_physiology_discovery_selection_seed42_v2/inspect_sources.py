"""Read-only unlabeled registry/source inventory; no feature selection or outcomes."""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import read, sha


def main():
    source = Path(r'E:\DRE-nips\new-pipeline\7-11')
    for name in ['ez_features.py', 'neuroez_c/evidence_views.py', 'raw_delta_encoder_seed42_v1/code/raw_physiological_features.py']:
        path = source / name
        print('SOURCE', name, path.exists(), sha(path) if path.exists() else None)
        if path.exists():
            text = path.read_text(encoding='utf-8-sig')
            if 'ez_features' in name:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = getattr(node, 'targets', [getattr(node, 'target', None)])
                        names = [x.id for x in targets if isinstance(x, ast.Name)]
                        if any('NAMES' in x or 'FEATURE' in x for x in names):
                            try:
                                print('DECLARATION', names, ast.literal_eval(node.value))
                            except (ValueError, TypeError):
                                pass
            else:
                print(text)
    root = Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
    for name in ['CANDIDATE_FEATURES_PRIVATE.pkl', 'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl', 'CANDIDATE_REGISTRY_PRIVATE.pkl']:
        path = root / name
        value = read(path)
        print('CACHE', name, sha(path), 'keys', list(value))
        print('SHAPES', {k: list(v.shape) for k, v in value.items() if hasattr(v, 'shape')})
    reg = read(Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl'))
    value = read(Path(next(iter(reg['files'].values()))['path']))
    print('OLD_WINDOW_KEYS', list(value))
    print('OLD_WINDOW_SHAPES', {k: list(v.shape) for k, v in value.items() if hasattr(v, 'shape')})
    rec = read(Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime\RECONSTRUCTION_PRIVATE.pkl'))
    print('RECONSTRUCTION_KEYS', list(rec))
    print('LEDGER_KEYS', list(rec['ledgers'][0]))
    first = rec['ledgers'][0]['records'][0]
    print('RECORD_KEYS', list(first))
    for key, value in first.items():
        if isinstance(value, dict):
            print('RECORD_NESTED_KEYS', key, list(value))


if __name__ == '__main__':
    main()
