"""Public-only checks; private patient metrics/bootstrap require the server verifier."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy',
           'mrr', 'hit_at_1', 'ez_fraction_bias']
SEEDS = [42, 43, 44]
ARMS = ['B0', 'PRUNED_REF', 'FULL', 'WO_FEATURE_PR', 'WO_RAW_PR',
        'WO_BOTH_PR', 'WO_GLOBAL_VIEW', 'WO_PRE_REFERENCE']
REFERENCES = {'B0': 'B0', 'PRUNED_REF': 'JOINT_PRUNED'}
CONTRASTS = {
    'FULL_MINUS_B0': {'FULL': 1, 'B0': -1},
    'FULL_MINUS_PRUNED_REF': {'FULL': 1, 'PRUNED_REF': -1},
    'FEATURE_PR_WITH_RAW': {'FULL': 1, 'WO_FEATURE_PR': -1},
    'RAW_PR_WITH_FEATURE': {'FULL': 1, 'WO_RAW_PR': -1},
    'BOTH_PR_JOINT': {'FULL': 1, 'WO_BOTH_PR': -1},
    'GLOBAL_VIEW': {'FULL': 1, 'WO_GLOBAL_VIEW': -1},
    'PRE_REFERENCE': {'FULL': 1, 'WO_PRE_REFERENCE': -1},
    'PR_INTERACTION': {'FULL': 1, 'WO_FEATURE_PR': -1,
                       'WO_RAW_PR': -1, 'WO_BOTH_PR': 1},
}
PRIMARY = ['FULL_MINUS_B0', 'FULL_MINUS_PRUNED_REF', 'FEATURE_PR_WITH_RAW',
           'RAW_PR_WITH_FEATURE', 'GLOBAL_VIEW', 'PRE_REFERENCE']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isfinite(float(a)) and abs(float(a) - float(b)) < 1e-12


def main(root):
    own = Path(__file__).resolve().parent
    protocol = read_json(own / 'PROTOCOL_LOCK.json')
    assert protocol['arms'] == ARMS and protocol['seeds'] == SEEDS
    assert protocol['references'] == REFERENCES
    assert protocol['data'] == dict(patients=80, channels=7635, seizures=256,
                                    valid_raw250_windows=1471965)

    def table(name):
        with (root / (name + '.csv')).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    def keyed(rows, fields):
        values = {tuple(row[k] for k in fields): row for row in rows}
        assert len(values) == len(rows), 'Duplicate aggregate keys'
        return values

    manifest = read_json(root / 'PACKAGE_MANIFEST.json')
    tree = ast.parse((own / 'package_public.py').read_text(encoding='utf-8'))
    allowed = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ALLOW'
                           for t in node.targets))
    assert len(allowed) == len(set(allowed)) == 30
    assert set(manifest['files']) == set(allowed)
    assert manifest['status'] == 'EXPLICIT_ANONYMOUS_ALLOWLIST'
    assert not manifest['private_artifacts_exported']
    forbidden = {'patient', 'patient_id', 'channel', 'channel_id',
                 'archive_indices', 'score_nez', 'prediction', 'probability', 'y', 'path'}
    for name, digest in manifest['files'].items():
        assert Path(name).name == name and not Path(name).is_absolute()
        assert '..' not in Path(name).parts
        p = root / name
        assert p.stat().st_size < 5000000 and sha(p) == digest
        if p.suffix == '.csv':
            with p.open(encoding='utf-8', newline='') as f:
                assert not set(next(csv.reader(f))) & forbidden

    admission = read_json(root / 'ADMISSION.json')
    assert admission['status'] == 'PASS' and admission['synthetic_tests'] == 6
    assert admission['reuse_cells'] == admission['initial_TRAIN_only_anchors'] == 30
    assert len(admission['source']) == 12
    assert all(sha(own / n) == h for n, h in admission['source'].items())
    assert admission['independent_full_logits_loss_all_gradients_oneAdam_exact']
    assert admission['maximum_wrapper_gradient_error'] == 0
    assert {r['arm'] for r in admission['new_path_admission']} == set(ARMS) - set(REFERENCES)
    assert all(r['complete_patient_backward_finite'] and r['no_formal_update']
               and r['parameters'] == 15473 for r in admission['new_path_admission'])
    push = read_json(root / 'SOURCE_PUSH_BEFORE_TRAIN.json')
    assert push['source_pushed_before_formal_fits'] and push['source'] == admission['source']
    assert len(push['commit']) == 40 and all(c in '0123456789abcdef' for c in push['commit'])

    expected = {(a, str(s), str(f)) for a in ARMS for s in SEEDS for f in range(1, 6)}
    reference_keys = {k for k in expected if k[0] in REFERENCES}
    reuse = keyed(table('ADMISSION_REUSE'), ['arm', 'seed', 'fold'])
    assert set(reuse) == reference_keys and all(r['stored_CAL_bit_exact'] == 'True' for r in reuse.values())
    assert all(r['origin_arm'] == REFERENCES[r['arm']] for r in reuse.values())
    anchors = keyed(table('ANCHOR_ADMISSION'), ['fold', 'seed', 'kind'])
    expected_anchors = {(str(f), str(s), k) for f in range(1, 6) for s in SEEDS for k in ['pre', 'no_pre']}
    assert set(anchors) == expected_anchors
    anchor_replay = keyed(table('ANCHOR_REPLAY'), ['fold', 'seed', 'kind'])
    assert set(anchor_replay) == expected_anchors
    assert all(r['initial_TRAIN_anchor_bit_exact'] == 'True' for r in anchor_replay.values())
    roles = keyed(table('ROLE_AUDIT'), ['fold'])
    assert set(roles) == {(str(f),) for f in range(1, 6)}
    assert all(int(r['cal_patients']) == int(r['assessment_patients']) == 8
               and 34 <= int(r['train_patients']) <= 36 for r in roles.values())
    for k, r in anchors.items():
        role = roles[(r['fold'],)]
        assert r['initial_only_not_current_reference'] == 'True'
        assert int(r['TRAIN_patients']) == int(role['train_patients'])
        assert int(r['TRAIN_channels']) == int(role['train_channels'])
        assert int(r['TRAIN_valid_windows']) == int(anchor_replay[k]['valid_windows']) > 0
        assert int(r['TRAIN_channels']) == int(anchor_replay[k]['channels'])

    status = read_json(root / 'TRAIN_STATUS.json')
    assert status['status'] == 'ALL120_COMPLETE' and status['fresh_fits'] == 90 and status['reused'] == 30
    cells = keyed(table('TRAIN_CELLS'), ['arm', 'seed', 'fold'])
    assert set(cells) == expected
    history = table('TRAIN_HISTORY')
    assert sum(r['reused'] == 'False' for r in history) == status['new_epochs']
    assert sum(r['reused'] == 'True' for r in history) == status['reused_epochs']
    for key, r in cells.items():
        assert (r['reused'] == 'True') == (r['arm'] in REFERENCES)
        assert int(r['parameters']) == protocol['parameters'][r['arm']]
        assert 1 <= int(r['epoch']) <= int(r['epochs']) <= 30 and int(r['epochs']) >= 6
        h = [q for q in history if (q['arm'], q['seed'], q['fold']) == key]
        assert len(h) == int(r['epochs'])
        assert sorted(int(q['epoch']) for q in h) == list(range(1, len(h) + 1))
        assert all((q['reused'] == 'True') == (r['arm'] in REFERENCES) for q in h)
        winner = max(h, key=lambda q: (float(q['macro_f1']), float(q['ez_f1']), -int(q['epoch'])))
        assert int(winner['epoch']) == int(r['epoch']) and close(winner['threshold'], r['threshold'])
        if key in reference_keys:
            assert reuse[key]['checkpoint_sha256'] == r['checkpoint_sha256']
            assert reuse[key]['calibration_sha256'] == r['calibration_sha256']

    coverage = keyed(table('COVERAGE'), ['arm', 'seed', 'fold'])
    replay = keyed(table('CALIBRATION_REPLAY'), ['arm', 'seed', 'fold'])
    assert set(coverage) == set(replay) == expected
    for key, r in coverage.items():
        role = roles[(r['fold'],)]
        assert int(r['patients']) == 8 and int(r['channels']) == int(role['assessment_channels'])
        assert int(r['valid_windows']) > 0 and (r['reused'] == 'True') == (r['arm'] in REFERENCES)
        assert replay[key]['CAL_logits_bit_exact'] == 'True'
        assert int(replay[key]['selected_epoch']) == int(cells[key]['epoch'])
        assert close(replay[key]['threshold'], cells[key]['threshold'])
    for f in range(1, 6):
        assert len({r['valid_windows'] for r in coverage.values() if r['fold'] == str(f)}) == 1
    selection = read_json(root / 'SELECTION_VERIFICATION.json')
    assert selection['status'] == 'PASS' and selection['cells'] == 120
    assert selection['initial_TRAIN_anchor_replays'] == 30 and selection['all_CAL_bit_exact']
    assert selection['TRAIN_only_preprocessing'] and selection['ASSESS_not_used_for_selection']
    freeze = read_json(root / 'FREEZE_STATUS.json')
    sealed = read_json(root / 'PREDICTION_STATUS.json')
    assert freeze['status'] == 'ALL120_FROZEN' and sealed['status'] == 'ALL120_SEALED_BEFORE_METRICS'
    assert freeze['freeze_sha256'] == sealed['freeze_sha256']

    mainrows = {r['arm']: r for r in table('MAIN_RESULTS')}
    seeds = keyed(table('SEED_RESULTS'), ['arm', 'seed'])
    folds = keyed(table('FOLD_RESULTS'), ['arm', 'seed', 'fold'])
    seed_sd = keyed(table('SEED_SD'), ['arm'])
    fold_sd = keyed(table('FOLD_SD'), ['arm', 'seed'])
    assert len(table('MAIN_RESULTS')) == len(mainrows) == 8 and set(mainrows) == set(ARMS)
    assert set(seeds) == {(a, str(s)) for a in ARMS for s in SEEDS} and set(folds) == expected
    assert set(seed_sd) == {(a,) for a in ARMS} and set(fold_sd) == set(seeds)
    metric_coverage = keyed(table('METRIC_COVERAGE'), ['arm', 'metric'])
    assert set(metric_coverage) == {(a, m) for a in ARMS for m in METRICS}
    for a in ARMS:
        assert int(mainrows[a]['unique_patients']) == 33 and int(mainrows[a]['patient_fold_seed_rows']) == 120
        assert int(mainrows[a]['seeds']) == 3
        sd = seed_sd[(a,)]
        for m in METRICS:
            values = [float(seeds[(a, str(s))][m]) for s in SEEDS]
            assert all(math.isfinite(v) for v in values)
            assert close(statistics.mean(values), mainrows[a][m]) and close(statistics.stdev(values), sd[m])
            assert int(metric_coverage[(a, m)]['unique_patients']) == int(metric_coverage[(a, m)]['finite_patients']) == 33
        for s in SEEDS:
            ss = fold_sd[(a, str(s))]
            ff = [folds[(a, str(s), str(f))] for f in range(1, 6)]
            assert all(int(r['patients']) == 8 for r in ff)
            assert all(close(statistics.stdev(float(r[m]) for r in ff), ss[m]) for m in METRICS)
        sources = [r for r in table('SOURCE_RESULTS') if r['arm'] == a]
        assert sum(int(r['unique_patients']) for r in sources) == 33
        assert len({r['center'] for r in sources}) == len(sources)
        for r in sources:
            suppressed = int(r['unique_patients']) < 5
            assert (r['suppressed'] == 'True') == suppressed
            assert all(math.isnan(float(r[m])) if suppressed else math.isfinite(float(r[m])) for m in METRICS)

    # These identities use aggregate tables only; they cannot recreate rankings
    # or pooled classification metrics from private per-channel predictions.
    pooled = keyed(table('POOLED_RESULTS'), ['arm', 'seed'])
    assert set(pooled) == set(seeds)
    expected_appearances = sum(int(r['assessment_channels']) for r in roles.values())
    for r in pooled.values():
        assert r['aggregation'] == 'POOLED_APPEARANCES_NOT_PRIMARY'
        assert int(r['channel_appearances']) == expected_appearances
        for m in ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy']:
            assert math.isfinite(float(r[m])) and 0 <= float(r[m]) <= 1
        assert math.isfinite(float(r['ez_fraction_bias'])) and -1 <= float(r['ez_fraction_bias']) <= 1
    for r in list(mainrows.values()) + list(seeds.values()) + list(folds.values()) + list(pooled.values()):
        assert close((float(r['f1']) + float(r['nez_f1'])) / 2, r['macro_f1'])

    ci = keyed(table('INTERVALS'), ['kind', 'contrast', 'metric'])
    family = keyed(table('PRIMARY_FAMILY_INTERVALS'), ['contrast', 'metric'])
    assert len(ci) == 144 and len(family) == 12
    assert set(ci) == ({('absolute', a, m) for a in ARMS for m in METRICS}
                       | {('paired', c, m) for c in CONTRASTS for m in METRICS})
    assert set(family) == {(c, m) for c in PRIMARY for m in ['auroc', 'macro_f1']}
    for (kind, contrast, metric), r in ci.items():
        weights = {contrast: 1} if kind == 'absolute' else CONTRASTS[contrast]
        assert close(sum(float(mainrows[a][metric]) * w for a, w in weights.items()), r['mean'])
        assert math.isfinite(float(r['low'])) and math.isfinite(float(r['high'])) and float(r['low']) <= float(r['high'])
        assert int(r['unique_patients']) == 33 and int(r['seeds']) == 3 and int(r['draws']) == 10000
    for (contrast, metric), r in family.items():
        assert close(sum(float(mainrows[a][metric]) * w for a, w in CONTRASTS[contrast].items()), r['mean'])
        assert int(r['comparisons']) == 12 and int(r['draws']) == 10000
        nominal = ci[('paired', contrast, metric)]
        assert math.isfinite(float(r['low'])) and math.isfinite(float(r['high']))
        assert float(r['low']) <= float(nominal['low']) <= float(nominal['high']) <= float(r['high'])

    evidence = {r['contrast']: r for r in table('COMPONENT_EVIDENCE')}
    assert len(table('COMPONENT_EVIDENCE')) == len(evidence) == 6 and set(evidence) == set(PRIMARY)
    points = {}
    for contrast, r in evidence.items():
        weights = CONTRASTS[contrast]
        point = all(sum(float(mainrows[a][m]) * w for a, w in weights.items()) > 0 for m in ['auroc', 'macro_f1'])
        count = sum(all(sum(float(seeds[(a, str(s))][m]) * w for a, w in weights.items()) > 0
                        for m in ['auroc', 'macro_f1']) for s in SEEDS)
        points[contrast] = point and count >= 2
        assert (r['mean_AUROC_and_Macro_both_positive'] == 'True') == point and int(r['seeds_both_positive']) == count
        assert (r['stable_positive_point_evidence'] == 'True') == points[contrast]
        assert (r['both_nominal_lowers_positive'] == 'True') == all(float(ci[('paired', contrast, m)]['low']) > 0 for m in ['auroc', 'macro_f1'])
        assert (r['both_family_lowers_positive'] == 'True') == all(float(family[(contrast, m)]['low']) > 0 for m in ['auroc', 'macro_f1'])
    delta = {m: float(mainrows['FULL'][m]) - float(mainrows['B0'][m]) for m in ['auroc', 'auprc', 'macro_f1', 'f1']}
    count = 0
    for s in SEEDS:
        d = {m: float(seeds[('FULL', str(s))][m]) - float(seeds[('B0', str(s))][m]) for m in delta}
        count += d['auroc'] > 0 and all(d[m] >= -.01 for m in ['auprc', 'macro_f1', 'f1'])
    performance = delta['auroc'] > 0 and all(delta[m] >= -.01 for m in ['auprc', 'macro_f1', 'f1']) and count >= 2
    gate = read_json(root / 'QUALIFICATION_GATE.json')
    expected_flags = {'performance_margin_pass': performance,
                      'both_separate_PR_point_rules_pass': points['FEATURE_PR_WITH_RAW'] and points['RAW_PR_WITH_FEATURE'],
                      'expanded_reference_point_rule_pass': points['FULL_MINUS_PRUNED_REF'],
                      'global_view_point_rule_pass': points['GLOBAL_VIEW'],
                      'pre_reference_point_rule_pass': points['PRE_REFERENCE']}
    assert all(gate[k] == v for k, v in expected_flags.items())
    qualified = all(expected_flags.values())
    assert gate['qualified_for_final_study_planning'] == qualified and gate['status'] == ('PASS' if qualified else 'FAIL')
    assert not gate['new_Outer_authorized'] and not gate['active_goal_achieved']

    audit = read_json(root / 'INDEPENDENT_VERIFICATION.json')
    assert audit['status'] == 'PASS' and audit['patient_fold_seed_rows'] == 960 and audit['patient_metric_checks'] == 8640
    assert audit['unique_patients'] == 33 and audit['nominal_interval_checks'] == 144 and audit['primary_family_interval_checks'] == 12
    assert audit['maximum_interval_discrepancy'] < 1e-12 and audit['all_roles_and_raw_window_coverage_verified']
    assert audit['no_new_Outer'] and not audit['active_goal_achieved']
    completed = read_json(root / 'COMPLETION_STATUS.json')
    assert completed['status'] == 'RAW_DUALVIEW_INTERNAL_COMPLETE_VERIFIED'
    assert completed['fresh_fits'] == 90 and completed['exact_reused_fits'] == 30
    assert completed['qualified_for_final_study_planning'] == qualified
    assert not completed['new_Outer'] and not completed['active_goal_achieved']
    reported = read_json(root / 'REPORT_STATUS.json')
    assert reported['status'] == 'ALL8_ARMS_3SEEDS_REPORTED'
    assert reported['patient_fold_seed_rows'] == 960 and reported['unique_patients'] == 33
    assert reported['nominal_intervals'] == 144 and reported['primary_family_intervals'] == 12
    assert reported['no_new_Outer'] and not reported['active_goal_achieved']
    print(json.dumps(dict(status='PASS', public_package_members=31, source_files=12,
                          arms=8, seeds=3, nominal_intervals=144, additional_intervals=12,
                          private_data_used=False, full_patient_bootstrap_recomputed=False,
                          private_independent_verifier_required=True)))


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results')
