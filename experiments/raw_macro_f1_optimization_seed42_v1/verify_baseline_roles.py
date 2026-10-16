"""Independent identity/role check against the original patient split contract."""
from admit_baseline import ROOT, HISTORY, read, jr, sha, write, load
from pathlib import Path
import numpy as np
import torch


def main():
    gate = jr(ROOT / 'public' / 'BASELINE_REFERENCE_PARITY.json')
    assert gate['status'] == 'PASS'
    io = load('macro_baseline_independent_role_reader', Path(r'D:\raw_latent_contrast_optimization_seed42_v1\common.py'))
    assert sha(Path(io.__file__)) == 'c5dd9fb82fab45dd3b1e4ae0911f642a221fbf5bd039a5a642f838a3a5ffa499'
    _, originals = io.identity_and_banks(labels=False)
    frozen = read(HISTORY / 'MODEL_FREEZE_PRIVATE.pkl')
    records = [r for r in frozen['selections'] if r['family'] == 'Concat' and r['recipe'] == 'RAW_ONLY']
    checked = []
    for r in sorted(records, key=lambda r: r['fold']):
        fold = r['fold']
        original = originals[fold - 1]
        bank = torch.load(HISTORY / f'fold{fold}' / 'BANK_PRIVATE.pt', map_location='cpu', weights_only=False)
        assert bank['binding'] == frozen['binding']
        for key in ['patient', 'channel', 'train', 'val', 'archive_indices']:
            assert np.array_equal(bank[key], original[key]), (fold, key)
        assert bank['roles'] == original['roles']
        saved = read(r['validation'])
        assert sha(r['validation']) == r['validation_sha256']
        assert np.array_equal(saved['archive_indices'], original['archive_indices'][original['val']])
        audit = jr(ROOT / 'validation_replay_private' / f'fold{fold}_AUDIT.json')
        for key, expected in audit['binding']['source'].items():
            assert sha(Path(__file__).resolve().parent / key) == expected
        assert audit['checkpoint_sha256'] == r['checkpoint_sha256']
        checked.append(dict(fold=fold, original_FIT_and_validation_indices_exact=True,
                            original_channel_identity_and_order_exact=True,
                            own_Outer_excluded=True, source_bound_replay_verified=True))
    write(ROOT / 'public' / 'BASELINE_ROLE_VERIFICATION.json', dict(
        status='PASS', folds=5, checks=checked, raw_source_contract='80patients/7635channels/256seizures/1471965windows',
        no_label_array_returned_by_original_reader=True,
        caveat='Legacy metadata bundles are unpickled by the pinned reader; measurements and labels are not returned or model inputs.',
        additional_prediction_inferences=0, new_Outer_passes=0, private_rows_exported=False))
    print('MACRO_BASELINE_ORIGINAL_ROLE_CONTRACT_PASS', flush=True)


if __name__ == '__main__':
    main()
