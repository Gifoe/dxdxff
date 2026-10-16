"""Validate the actual strong Raw-only reference, without new Outer inference."""
import gc
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from pure_model import PureRaw

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\raw_macro_f1_optimization_seed42_v1_runtime')
HISTORY = Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
SOURCE = Path(r'D:\m8_physiology_discovery_selection_seed42_v2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for q in iter(lambda: f.read(8 << 20), b''):
            h.update(q)
    return h.hexdigest()


def jr(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read(path):
    path = Path(path)
    assert sha(path) == jr(path.with_suffix('.sha256.json'))['sha256']
    with path.open('rb') as f:
        return pickle.load(f)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    lock = jr(HERE / 'BASELINE_REPLAY_LOCK.json')
    io = load('macro_reference_raw_only_io', Path(r'D:\raw_latent_contrast_optimization_seed42_v1\common.py'))
    assert sha(Path(io.__file__)) == 'c5dd9fb82fab45dd3b1e4ae0911f642a221fbf5bd039a5a642f838a3a5ffa499'
    assert sha(io.BASE / 'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl') == io.EXPECTED_REG
    assert sha(io.BASE / 'RECONSTRUCTION_PRIVATE.pkl') == io.EXPECTED_REC
    status = jr(HISTORY / 'public' / 'NEURAL_STATUS.json')
    assert status['status'] == 'ALL80_LOGICAL_MODELS_FROZEN'
    assert status['model_freeze_sha256'] == sha(HISTORY / 'MODEL_FREEZE_PRIVATE.pkl')
    frozen = read(HISTORY / 'MODEL_FREEZE_PRIVATE.pkl')
    for name, expected in frozen['binding']['source'].items():
        assert sha(SOURCE / name) == expected
    assert frozen['binding']['torch'] == torch.__version__
    assert frozen['binding']['numpy'] == np.__version__
    rows = [r for r in frozen['selections'] if r['family'] == 'Concat' and r['recipe'] == 'RAW_ONLY']
    assert len(rows) == 5 and {r['fold'] for r in rows} == set(range(1, 6))
    binding = dict(source={n: sha(HERE / n) for n in
                          ['TARGET_DEFINITION.json', 'BASELINE_REPLAY_LOCK.json', 'pure_model.py', 'admit_baseline.py']},
                   historical_freeze_sha256=sha(HISTORY / 'MODEL_FREEZE_PRIVATE.pkl'),
                   raw_registry_sha256=sha(io.BASE / 'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),
                   reconstruction_sha256=sha(io.BASE / 'RECONSTRUCTION_PRIVATE.pkl'),
                   reference_rows=[{k: r[k] for k in ['fold', 'checkpoint_sha256', 'validation_sha256', 'threshold', 'epoch']} for r in rows])
    input_lock = ROOT / 'BASELINE_INPUT_FREEZE_PRIVATE.json'
    if input_lock.exists():
        assert jr(input_lock) == binding
    else:
        write(input_lock, binding)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cudnn.deterministic
    store = io.RawStore()
    reports, validation_ids = [], set()
    for r in sorted(rows, key=lambda r: r['fold']):
        fold = r['fold']
        assert r['groups'] == [] and sha(r['checkpoint']) == r['checkpoint_sha256']
        assert sha(r['validation']) == r['validation_sha256']
        cp = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['epoch'] == r['epoch'] and cp['threshold'] == r['threshold']
        bank = torch.load(HISTORY / f'fold{fold}' / 'BANK_PRIVATE.pt', map_location='cpu', weights_only=False)
        # Historical bundles contain measurements and labels. Retain identity/
        # validation-role metadata only; no measurement/label is passed to a model.
        d = {k: bank[k] for k in ['patient', 'channel', 'val', 'archive_indices', 'roles']}
        assert set(d['patient']).isdisjoint(d['roles']['test'])
        assert set(d['patient'][d['val']]) == d['roles']['validation']
        del bank
        saved = read(r['validation'])
        assert saved['checkpoint_sha256'] == r['checkpoint_sha256']
        assert np.array_equal(saved['archive_indices'], d['archive_indices'][d['val']])
        models = [PureRaw(compact=False).cuda().eval(), PureRaw(compact=True).cuda().eval()]
        for m in models:
            m.load_reference(cp['model'])
            m.requires_grad_(False)
        scores = [np.empty(len(d['val']), np.float32) for _ in models]
        patients = d['patient'][d['val']]
        with torch.inference_mode():
            for pid in sorted(set(patients)):
                loc = np.flatnonzero(patients == pid)
                ii = d['val'][loc]
                raw, mask, times = store.get(pid, d['channel'][ii])
                for k, m in enumerate(models):
                    scores[k][loc] = m(raw, mask, times).cpu().numpy()
        maximum, probability_error, disagreement = [], [], []
        reference_prob = torch.sigmoid(torch.from_numpy(saved['logit'])).numpy()
        for score in scores:
            assert np.isfinite(score).all()
            error = float(np.max(np.abs(score - saved['logit'])))
            prob = torch.sigmoid(torch.from_numpy(score)).numpy()
            pe = float(np.max(np.abs(prob - reference_prob)))
            changes = int(np.count_nonzero((prob >= r['threshold']) != (reference_prob >= r['threshold'])))
            assert error <= lock['maximum_logit_error'] and pe <= lock['maximum_NEZ_probability_error']
            assert changes == lock['maximum_threshold_prediction_disagreements']
            maximum.append(error); probability_error.append(pe); disagreement.append(changes)
        report = dict(fold=fold, validation_patients=len(set(patients)), channels=len(patients),
                      reference160_logit_max_error=maximum[0], compact96_logit_max_error=maximum[1],
                      probability_max_errors=probability_error, threshold_prediction_disagreements=disagreement,
                      checkpoint_sha256=r['checkpoint_sha256'], threshold_unchanged=True,
                      only_Raw_mask_original_times_model_inputs=True)
        write(ROOT / 'validation_replay_private' / f'fold{fold}_AUDIT.json', dict(binding=binding, **report))
        reports.append(report)
        validation_ids.update(map(str, set(patients)))
        print('MACRO_REFERENCE_VALIDATION_PARITY_PASS', fold, maximum, flush=True)
        del models, cp, d, scores, raw, mask, times
        gc.collect(); torch.cuda.empty_cache()
    assert sum(r['validation_patients'] for r in reports) == 65 and len(validation_ids) == 47
    write(ROOT / 'public' / 'BASELINE_REFERENCE_PARITY.json', dict(
        status='PASS', folds=5, validation_patient_appearances=65, unique_validation_patients=47,
        input_binding_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
        actual_frozen_checkpoint_replay=True, reports=reports, no_new_training=True, new_Outer_passes=0,
        primary_target='patient-equal Macro-F1>0.65', default_head='reference160_with64literal_zero_slots',
        private_predictions_or_identities_exported=False))
    print('MACRO_RAW_REFERENCE_ADMISSION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
