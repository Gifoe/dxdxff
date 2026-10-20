"""Independent release-only checks; never loads data, predictions or models.

Not part of scientific execution and never changes training/evaluation. Verify a
strict server-generated anonymous ZIP before extracting it into the repository.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np

ARMS = ['B0_ORIGINAL', 'FULL', 'WO_FEATURE_PR', 'WO_RAW_PR', 'WO_BOTH_PR',
        'WO_WINDOW_PR', 'WO_EARLY_FILM', 'WO_LATE_GATE', 'WO_PRE_REFERENCE']
METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1', 'ez_fraction_bias']
ALLOW = ['ADMISSION.json', 'ROLE_AUDIT.csv', 'PREPROCESSING_ADMISSION.csv', 'ADMISSION_REUSE.csv',
         'SOURCE_PUSH_BEFORE_TRAIN.json', 'TRAIN_STATUS.json', 'TRAIN_CELLS.csv', 'TRAIN_HISTORY.csv',
         'INTERNAL_VERIFICATION.json', 'VALIDATION_REPLAY_CHECKS.csv', 'PREPROCESSING_REPLAY.csv',
         'NULL_CONTROL_RETRAIN_PARITY.csv', 'FREEZE_STATUS.json', 'PREDICTION_STATUS.json',
         'RAW_COVERAGE.csv', 'MAIN_RESULTS.csv', 'FOLD_RESULTS.csv', 'FOLD_SD.csv', 'SOURCE_RESULTS.csv',
         'METRIC_COVERAGE.csv', 'INTERVALS.csv', 'PRIMARY_FAMILY_INTERVALS.csv', 'COMPONENT_EVIDENCE.csv',
         'GOAL_AUDIT.json', 'POOLED_RESULTS.csv', 'REPORT_STATUS.json', 'INDEPENDENT_OUTER_VERIFICATION.json',
         'OUTER_STATUS.json', 'REPORT.md']
HERE = Path(__file__).resolve().parent
PRIMARY = ['FULL_MINUS_B0', 'FEATURE_PR', 'RAW_PR', 'WINDOW_PR', 'EARLY_FILM', 'LATE_GATE', 'PRE_REFERENCE']
WEIGHTS = {'FULL_MINUS_B0': {'FULL': 1, 'B0_ORIGINAL': -1},
           'FEATURE_PR': {'FULL': 1, 'WO_FEATURE_PR': -1}, 'RAW_PR': {'FULL': 1, 'WO_RAW_PR': -1},
           'BOTH_PR': {'FULL': 1, 'WO_BOTH_PR': -1}, 'WINDOW_PR': {'FULL': 1, 'WO_WINDOW_PR': -1},
           'EARLY_FILM': {'FULL': 1, 'WO_EARLY_FILM': -1}, 'LATE_GATE': {'FULL': 1, 'WO_LATE_GATE': -1},
           'PRE_REFERENCE': {'FULL': 1, 'WO_PRE_REFERENCE': -1},
           'PR_INTERACTION': {'FULL': 1, 'WO_FEATURE_PR': -1, 'WO_RAW_PR': -1, 'WO_BOTH_PR': 1}}


def validate_archive(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == 30
        assert set(names) == set(ALLOW) | {'PACKAGE_MANIFEST.json'}
        assert all('/' not in n and '\\' not in n and ':' not in n and '..' not in n for n in names)
        assert all(i.file_size < 5000000 and not i.flag_bits & 1 for i in z.infolist())
        assert sum(i.file_size for i in z.infolist()) < 15000000
        data = {n: z.read(n) for n in names}
    manifest = json.loads(data.pop('PACKAGE_MANIFEST.json'))
    assert manifest['status'] == 'EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST'
    assert manifest['private_artifacts_exported'] is False and set(manifest['files']) == set(ALLOW)
    assert all(hashlib.sha256(data[n]).hexdigest() == h for n, h in manifest['files'].items())
    forbidden = {'patient', 'patient_id', 'channel', 'channel_id', 'archive_indices', 'prediction', 'score_nez', 'y', 'path'}
    for n, b in data.items():
        text = b.decode('utf-8')
        assert not any(s in text for s in ['password=', 'PRIVATE KEY', 'fit_patient_ids'])
        if n.endswith('.csv'):
            assert not set(next(csv.reader(io.StringIO(text)))) & forbidden
    return data


def check(path):
    data = validate_archive(path)
    def obj(n):
        return json.loads(data[n+'.json'])
    def tab(n):
        return list(csv.DictReader(io.StringIO(data[n+'.csv'].decode('utf-8'))))
    admission = obj('ADMISSION')
    assert admission['status'] == 'PASS' and admission['exact_original_reference_replays'] == 5
    assert admission['unique_math_model_reporting_tests'] == 9
    assert admission['unchanged_noPR_logits_loss_gradients_oneAdam_rng_bit_exact'] is True
    assert len(admission['source']) == 17 and all(hashlib.sha256((HERE/n).read_bytes()).hexdigest() == h for n, h in admission['source'].items())
    pushed = obj('SOURCE_PUSH_BEFORE_TRAIN')
    assert pushed['source'] == admission['source'] and pushed['source_pushed_before_formal_fits'] is True
    assert len(pushed['commit']) == 40 and all(c in '0123456789abcdef' for c in pushed['commit'])
    for name, status in [('TRAIN_STATUS', 'ALL45_MODEL_CELLS_FROZEN'), ('INTERNAL_VERIFICATION', 'PASS'),
                         ('FREEZE_STATUS', 'ALL45_FROZEN_BEFORE_NEW_OUTER'), ('PREDICTION_STATUS', 'ALL45_OUTER_CELLS_SEALED'),
                         ('REPORT_STATUS', 'ALL9_FIXED_GROUPS_REPORTED'), ('INDEPENDENT_OUTER_VERIFICATION', 'PASS'),
                         ('OUTER_STATUS', 'COMPLETE_VERIFIED_RGF0_NINE_GROUP_FIVE_FOLD_OUTER')]:
        assert obj(name)['status'] == status
    assert obj('TRAIN_STATUS')['actual_new_fits'] == obj('INTERNAL_VERIFICATION')['actual_new_fits'] == 40
    checkdata = obj('INDEPENDENT_OUTER_VERIFICATION')
    assert checkdata['patient_rows'] == 720 and checkdata['patient_metric_checks'] == 6480
    assert checkdata['nominal_interval_checks'] == 162 and checkdata['primary_family_interval_checks'] == 14
    assert checkdata['maximum_interval_difference'] < 1e-12
    assert checkdata['fixed_goal_audit_independently_recomputed'] is True
    cells = tab('TRAIN_CELLS')
    assert len(cells) == len({(r['arm'], r['fold']) for r in cells}) == 45
    assert sum(r['reused'] == 'True' for r in cells) == 5
    for r in cells:
        assert r['arm'] in ARMS and 1 <= int(r['epoch']) <= int(r['epochs']) <= 30 and int(r['epochs']) >= 6
        assert (r['reused'] == 'True') == (r['arm'] == 'B0_ORIGINAL')
        assert int(r['parameters']) == (10377 if r['arm'] in ['WO_EARLY_FILM', 'WO_LATE_GATE'] else 11425)
    for name in ['VALIDATION_REPLAY_CHECKS', 'RAW_COVERAGE', 'FOLD_RESULTS']:
        assert len(tab(name)) == 45
    assert len(tab('NULL_CONTROL_RETRAIN_PARITY')) == len(tab('PREPROCESSING_REPLAY')) == 5
    for r in tab('NULL_CONTROL_RETRAIN_PARITY'):
        assert all(v == 'True' for k, v in r.items() if k != 'fold')
    raw = tab('RAW_COVERAGE')
    for arm in ARMS:
        rows = [r for r in raw if r['arm'] == arm]
        assert len(rows) == 5 and sum(int(r['patients']) for r in rows) == 80
        assert sum(int(r['channels']) for r in rows) == 7635 and sum(int(r['valid_windows']) for r in rows) == 1471965
    main = {r['arm']: r for r in tab('MAIN_RESULTS')}
    assert set(main) == set(ARMS) and len(tab('MAIN_RESULTS')) == 9
    assert all(int(r['patients']) == 80 and all(np.isfinite(float(r[m])) for m in METRICS) for r in main.values())
    assert abs(float(main['B0_ORIGINAL']['macro_f1'])-.6538278218104221) < 1e-12
    assert abs(float(main['B0_ORIGINAL']['auroc'])-.7522551973793955) < 1e-12
    nominal, family = tab('INTERVALS'), tab('PRIMARY_FAMILY_INTERVALS')
    assert len(nominal) == 162 and len(family) == 14
    assert len({(r['kind'], r['contrast'], r['metric']) for r in nominal}) == 162
    for r in nominal:
        w = {r['contrast']: 1} if r['kind'] == 'absolute' else WEIGHTS[r['contrast']]
        assert r['metric'] in METRICS and int(r['patients']) == 80 and int(r['draws']) == 10000
        mean = sum(float(main[a][r['metric']])*coefficient for a, coefficient in w.items())
        assert abs(float(r['mean'])-mean) < 1e-12 and float(r['low']) <= float(r['high'])
    assert {(r['contrast'], r['metric']) for r in family} == {(c, m) for c in PRIMARY for m in ['auroc', 'macro_f1']}
    assert all(int(r['comparisons']) == 14 and int(r['draws']) == 10000 and abs(float(r['tail'])-.05/28) < 1e-15 for r in family)
    full, b0 = main['FULL'], main['B0_ORIGINAL']
    components = tab('COMPONENT_EVIDENCE')
    assert len(components) == 7 and {r['contrast'] for r in components} == set(PRIMARY)
    goal = obj('GOAL_AUDIT')
    expected = dict(main_macro_f1_above_0p64=float(full['macro_f1']) > .64,
                    macro_f1_within_1pp_of_original=float(full['macro_f1'])-float(b0['macro_f1']) >= -.01,
                    main_auroc_above_original=float(full['auroc']) > float(b0['auroc']),
                    all_retained_components_have_stable_two_metric_evidence=all(
                        r['stable_within_this_seed_protocol'] == 'True' for r in components if r['contrast'] != 'FULL_MINUS_B0'))
    assert goal['conditions'] == expected and goal['status'] == ('PASS' if all(expected.values()) else 'FAIL')
    assert goal['selected_arm'] is None and goal['outer_is_exploratory_repeated_test'] is True
    assert obj('OUTER_STATUS')['independent_confirmation'] is False
    assert obj('OUTER_STATUS')['selected_arm'] is None
    return dict(status='PUBLIC_RELEASE_PASS', private_artifacts_loaded=False, files=30,
                patient_rows_independently_verified_on_server=720, metric_values_independently_verified_on_server=6480,
                actual_goal_status=goal['status'], source_files_unchanged=17,
                package_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def self_test():
    payload = {n: (b'field,value\ncount,1\n' if n.endswith('.csv') else b'{}\n') for n in ALLOW}
    manifest = dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',
                    files={n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}, private_artifacts_exported=False)
    def make(path, content, listed, extra=None):
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for n, value in content.items():
                z.writestr(n, value)
            z.writestr('PACKAGE_MANIFEST.json', json.dumps(listed))
            if extra:
                z.writestr(*extra)
    with tempfile.TemporaryDirectory(prefix='rgf0_release_validation_') as directory:
        path = Path(directory)/'synthetic_validator_input.zip'
        make(path, payload, manifest)
        assert len(validate_archive(path)) == 29
        failures = []
        bad = dict(payload, **{'REPORT.md': b'corrupt byte sequence'})
        failures.append((bad, manifest, None))
        moved = dict(payload)
        moved['../REPORT.md'] = moved.pop('REPORT.md')
        failures.append((moved, manifest, None))
        for name, value in [('MAIN_RESULTS.csv', b'patient_id,value\nsynthetic,1\n'), ('REPORT.md', b'PRIVATE KEY')]:
            bad = dict(payload, **{name: value})
            listed = dict(manifest, files={n: hashlib.sha256(v).hexdigest() for n, v in bad.items()})
            failures.append((bad, listed, None))
        failures.append((payload, dict(manifest, private_artifacts_exported=True), None))
        failures.append((payload, manifest, ('unapproved.bin', b'not allowed')))
        for content, listed, extra in failures:
            make(path, content, listed, extra)
            try:
                validate_archive(path)
            except AssertionError:
                continue
            raise AssertionError('An unsafe or corrupted synthetic archive was accepted')
    return dict(status='RELEASE_VALIDATOR_SELF_TEST_PASS', cases=7, benchmark_metrics_generated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('zip_path', type=Path, nargs='?')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    assert args.self_test or args.zip_path is not None
    print(json.dumps(self_test() if args.self_test else check(args.zip_path), indent=2, sort_keys=True))
