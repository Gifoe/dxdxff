"""Reporting-only precision correction; never fits models or generates predictions.

Runs the frozen independent verifier with a single in-memory correction: promote
stored probabilities to float64 before sklearn's probability metrics/clipping.
The scientific metric engine already uses this dtype and 1e-15 clipping.
"""
import sys
from pathlib import Path

CODE = Path(r'D:\m8_raw_backbones_pr_seed42_v1')
sys.path.insert(0, str(CODE))
from support import *
from sklearn.metrics import log_loss

VERIFIER_SHA256 = '122058cdc00c197b1d52ce7990ccd4cb1f015c0fd279bff35942ac23f360344a'


def run():
    verifier = CODE / 'tests/verify_results.py'
    assert sha(verifier) == VERIFIER_SHA256
    original = verifier.read_text(encoding='utf-8')
    before = 'p=g.probability.to_numpy();pred='
    after = 'p=g.probability.to_numpy(dtype=np.float64);pred='
    assert original.count(before) == 1
    lock = jr(CODE / 'configs/PROTOCOL_LOCK.json')
    protected = [RUNTIME / 'INPUT_FREEZE_PRIVATE.json']
    for model in lock['models']:
        protected += [RUNTIME/model/f for f in (
            'PREDICTIONS_PRIVATE.pkl', 'PATIENT_METRICS_PRIVATE.pkl',
            'FROZEN_BEFORE_OUTER_PRIVATE.json', 'OUTER_PREDICTIONS_SEAL_PRIVATE.json')]
        for selection in jr(RUNTIME/model/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selections']:
            protected.append(Path(selection['checkpoint']))
    hashes = {str(p): sha(p) for p in protected}
    dtype_counts = {}
    old_errors, corrected_errors = [], []
    for model in lock['models']:
        frame = read(RUNTIME/model/'PREDICTIONS_PRIVATE.pkl')
        rows = read(RUNTIME/model/'PATIENT_METRICS_PRIVATE.pkl').set_index(['version','fold','patient'])
        for key, g in frame.groupby(['version','fold','patient']):
            p = g.probability.to_numpy()
            dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + 1
            y = g.y.to_numpy()
            target = rows.loc[key, 'log_loss']
            old = log_loss(y, np.clip(p, 1e-15, 1-1e-15))
            corrected = log_loss(y, np.clip(p.astype(np.float64), 1e-15, 1-1e-15))
            old_errors.append(abs(old-target))
            corrected_errors.append(abs(corrected-target))
    assert len(corrected_errors) == 640 and max(corrected_errors) <= 1e-12
    namespace = {'__name__': 'reporting_precision_corrected_frozen_verifier', '__file__': str(verifier)}
    exec(compile(original.replace(before, after), str(verifier)+'[float64-reporting-only]', 'exec'), namespace)
    namespace['run'](RUNTIME)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    assert sha(verifier) == VERIFIER_SHA256
    pub = RUNTIME/'public'
    write(pub/'REPORTING_VERIFIER_AUDIT.json', dict(
        status='PASS', original_verifier_sha256=VERIFIER_SHA256,
        reporting_wrapper_sha256=sha(Path(__file__)), patient_rows=640,
        original_probability_dtypes=dtype_counts,
        original_sklearn_log_loss_max_error=float(max(old_errors)),
        float64_sklearn_log_loss_max_error=float(max(corrected_errors)),
        numerical_tolerance=1e-12, tolerance_unchanged=True,
        correction='float64 probabilities before sklearn metrics and 1e-15 clipping, matching frozen metric engine',
        training_or_inference_run=False, predictions_changed=False, thresholds_changed=False,
        frozen_sources_changed=False, all_protected_artifact_hashes_unchanged=True,
        protected_artifact_count=len(hashes)))
    write(pub/'PUBLIC_SHA256.json', {p.name: sha(p) for p in sorted(pub.iterdir())
                                   if p.is_file() and p.name != 'PUBLIC_SHA256.json'})
    print('REPORTING_ONLY_PRECISION_VERIFICATION_PASS', flush=True)


if __name__ == '__main__':
    run()
