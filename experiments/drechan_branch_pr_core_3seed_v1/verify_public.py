"""Public-only release checks; not a replacement for private patient verification.

Uses only the Python standard library. Never imports the training dependency tree,
opens a checkpoint, or recalculates endpoints from patient/channel predictions.
"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1', 'ez_fraction_bias']
ARMS = ['FULL', 'WO_EARLY_FILM', 'WO_LATE_GATE', 'WO_BOTH_GUIDANCE', 'WO_RAW_PRE_REFERENCE', 'EARLY_FILM_MISMATCH']
SEEDS = [42, 43, 44]
CONTRASTS = {
    'EARLY_FILM': {'FULL': 1, 'WO_EARLY_FILM': -1},
    'LATE_GATE': {'FULL': 1, 'WO_LATE_GATE': -1},
    'BOTH_GUIDANCE': {'FULL': 1, 'WO_BOTH_GUIDANCE': -1},
    'PRE_REFERENCE': {'FULL': 1, 'WO_RAW_PRE_REFERENCE': -1},
    'CORRECT_EARLY_CONTEXT': {'FULL': 1, 'EARLY_FILM_MISMATCH': -1},
    'GUIDANCE_INTERACTION': {'FULL': 1, 'WO_EARLY_FILM': -1, 'WO_LATE_GATE': -1, 'WO_BOTH_GUIDANCE': 1},
}
PRIMARY = ['EARLY_FILM', 'LATE_GATE', 'PRE_REFERENCE', 'CORRECT_EARLY_CONTEXT']
PRIMARY_METRICS = ['auroc', 'macro_f1']


def public_allowlist(source):
    tree = ast.parse(source)
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ALLOW' for t in node.targets)]
    assert len(values) == 1 and len(values[0]) == len(set(values[0]))
    return set(values[0])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) < 1e-12


def main(root):
    own = Path(__file__).resolve().parent
    parent = own.parent / 'drechan_branch_pr_residual_3seed_v1'

    def read(name):
        return json.loads((root / (name + '.json')).read_text(encoding='utf-8'))

    def table(name):
        with (root / (name + '.csv')).open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))

    manifest = read('PACKAGE_MANIFEST')
    allowed = public_allowlist((own / 'package_public.py').read_text(encoding='utf-8'))
    assert manifest['status'] == 'EXPLICIT_ANONYMOUS_ALLOWLIST' and not manifest['private_artifacts_exported']
    assert set(manifest['files']) == allowed
    forbidden = {'patient', 'patient_id', 'channel', 'channel_id', 'archive_indices', 'prediction', 'probability', 'score_nez', 'y', 'path'}
    for name, digest in manifest['files'].items():
        assert Path(name).name == name and not Path(name).is_absolute() and '..' not in Path(name).parts
        path = root / name
        assert path.stat().st_size < 5_000_000 and sha(path) == digest
        if path.suffix == '.csv':
            with path.open(newline='', encoding='utf-8') as handle:
                assert not set(next(csv.reader(handle))) & forbidden

    admission = read('ADMISSION')
    assert admission['status'] == 'PASS' and len(admission['source']) == 12 and len(admission['parent_source']) == 13
    assert admission['initial_bridges'] == 15 and admission['full_logits_loss_all_gradients_oneAdam_exact']
    assert admission['maximum_wrapper_gradient_error'] == 0 and admission['no_assessment_used'] and admission['no_new_Outer']
    assert all(sha(own / name) == digest for name, digest in admission['source'].items())
    assert all(sha(parent / name) == digest for name, digest in admission['parent_source'].items())
    protocol = json.loads((own / 'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    assert protocol['arms'] == ARMS and protocol['seeds'] == SEEDS
    assert read('SOURCE_PUSH_BEFORE_TRAIN')['source'] == admission['source']

    audit = read('INDEPENDENT_VERIFICATION')
    assert audit['status'] == 'PASS' and audit['patient_fold_seed_rows'] == 720 and audit['patient_metric_checks'] == 6480
    assert audit['unique_patients'] == 33 and audit['nominal_interval_checks'] == 108 and audit['primary_family_interval_checks'] == 8
    assert audit['maximum_interval_discrepancy'] < 1e-12 and audit['all_roles_and_raw_window_coverage_verified']
    assert audit['no_new_Outer'] and not audit['active_goal_achieved']
    assert read('NEW_TRAIN_STATUS')['status'] == 'ALL75_NEW_COMPLETE'
    train = read('TRAIN_STATUS')
    assert train['status'] == 'ALL90_COMPLETE' and train['fresh_fits'] == 75 and train['reused'] == 15
    expected = {(a, s, f) for a in ARMS for s in SEEDS for f in range(1, 6)}
    cells = table('TRAIN_CELLS')
    assert len(cells) == 90 and {(r['arm'], int(r['seed']), int(r['fold'])) for r in cells} == expected
    for row in cells:
        assert (row['reused'] == 'True') == (row['arm'] == 'FULL')
        assert 1 <= int(row['epoch']) <= int(row['epochs']) <= 30 and int(row['epochs']) >= 6
        assert int(row['parameters']) == protocol['parameters'][row['arm']]
    history = table('TRAIN_HISTORY')
    assert sum(r['reused'] == 'False' for r in history) == train['new_epochs']
    assert sum(r['reused'] == 'True' for r in history) == train['reused_epochs']
    for row in cells:
        rows = [r for r in history if (r['arm'], r['seed'], r['fold']) == (row['arm'], row['seed'], row['fold'])]
        assert len(rows) == int(row['epochs']) and sorted(int(r['epoch']) for r in rows) == list(range(1, len(rows) + 1))
    selection = read('SELECTION_VERIFICATION')
    assert selection['status'] == 'PASS' and selection['cells'] == 90 and selection['all_CAL_bit_exact']
    assert selection['parent_initial_TRAIN_anchor_replays'] == 15 and selection['ASSESS_not_used_for_selection']
    assert len(table('CALIBRATION_REPLAY')) == 90
    assert read('FREEZE_STATUS')['status'] == 'ALL90_FROZEN'
    sealed = read('PREDICTION_STATUS')
    assert sealed['status'] == 'ALL90_SEALED_BEFORE_METRICS' and sealed['freeze_sha256'] == read('FREEZE_STATUS')['freeze_sha256']
    coverage = table('COVERAGE')
    assert len(coverage) == 90 and {(r['arm'], int(r['seed']), int(r['fold'])) for r in coverage} == expected
    for row in coverage:
        assert int(row['patients']) == 8 and int(row['channels']) > 0 and int(row['valid_windows']) > 0
        assert (row['reused'] == 'True') == (row['arm'] == 'FULL')
        if row['arm'] == 'EARLY_FILM_MISMATCH':
            assert int(row['changed_valid_windows']) == int(row['eligible_valid_windows'])
            assert int(row['changed_valid_windows']) + int(row['singleton_valid_windows']) == int(row['total_valid_windows']) == int(row['valid_windows'])

    mainrows = {r['arm']: r for r in table('MAIN_RESULTS')}
    seedrows = {(r['arm'], int(r['seed'])): r for r in table('SEED_RESULTS')}
    assert len(mainrows) == 6 and set(mainrows) == set(ARMS) and len(seedrows) == 18
    assert len(table('FOLD_RESULTS')) == 90 and len(table('FOLD_SD')) == 18 and len(table('SEED_SD')) == 6
    for arm in ARMS:
        assert int(mainrows[arm]['unique_patients']) == 33 and int(mainrows[arm]['patient_fold_seed_rows']) == 120
        sd = next(r for r in table('SEED_SD') if r['arm'] == arm)
        for metric in METRICS:
            values = [float(seedrows[(arm, seed)][metric]) for seed in SEEDS]
            close(statistics.mean(values), mainrows[arm][metric])
            close(statistics.stdev(values), sd[metric])
        for seed in SEEDS:
            folds = [r for r in table('FOLD_RESULTS') if r['arm'] == arm and int(r['seed']) == seed]
            assert len(folds) == 5 and {int(r['fold']) for r in folds} == set(range(1, 6))
            sd = next(r for r in table('FOLD_SD') if r['arm'] == arm and int(r['seed']) == seed)
            for metric in METRICS:
                close(statistics.stdev(float(r[metric]) for r in folds), sd[metric])
        sources = [r for r in table('SOURCE_RESULTS') if r['arm'] == arm]
        assert sum(int(r['unique_patients']) for r in sources) == 33
        for row in sources:
            suppressed = int(row['unique_patients']) < 5
            assert (row['suppressed'] == 'True') == suppressed
            assert all(math.isnan(float(row[m])) if suppressed else math.isfinite(float(row[m])) for m in METRICS)
    mc = table('METRIC_COVERAGE')
    assert len(mc) == 54 and {(r['arm'], r['metric']) for r in mc} == {(a, m) for a in ARMS for m in METRICS}
    assert all(int(r['unique_patients']) == int(r['finite_patients']) == 33 for r in mc)
    with (parent / 'results/MAIN_RESULTS.csv').open(newline='', encoding='utf-8') as handle:
        parent_full = next(r for r in csv.DictReader(handle) if r['arm'] == 'FULL')
    for metric in METRICS:
        close(mainrows['FULL'][metric], parent_full[metric])

    ci, family = table('INTERVALS'), table('PRIMARY_FAMILY_INTERVALS')
    assert len(ci) == 108 and len(family) == 8
    assert {(r['contrast'], r['metric']) for r in family} == {(c, m) for c in PRIMARY for m in PRIMARY_METRICS}
    assert {(r['kind'], r['contrast'], r['metric']) for r in ci} == ({('absolute', a, m) for a in ARMS for m in METRICS} | {('paired', c, m) for c in CONTRASTS for m in METRICS})
    for row in ci + family:
        weights = {row['contrast']: 1} if row.get('kind') == 'absolute' else CONTRASTS[row['contrast']]
        close(sum(float(mainrows[a][row['metric']]) * w for a, w in weights.items()), row['mean'])
        assert math.isfinite(float(row['low'])) and math.isfinite(float(row['high'])) and float(row['low']) <= float(row['high'])
        assert int(row['draws']) == 10000
        if 'kind' in row:
            assert int(row['unique_patients']) == 33 and int(row['seeds']) == 3
        else:
            assert int(row['comparisons']) == 8
    evidence = table('CORE_COMPONENT_EVIDENCE')
    assert len(evidence) == 4 and {r['contrast'] for r in evidence} == set(PRIMARY)
    checks = {}
    for row in evidence:
        weights = CONTRASTS[row['contrast']]
        positive = all(sum(float(mainrows[a][m]) * w for a, w in weights.items()) > 0 for m in PRIMARY_METRICS)
        count = sum(all(sum(float(seedrows[(a, seed)][m]) * w for a, w in weights.items()) > 0 for m in PRIMARY_METRICS) for seed in SEEDS)
        assert (row['mean_AUROC_and_Macro_both_positive'] == 'True') == positive and int(row['seeds_both_positive']) == count
        checks[row['contrast']] = positive and count >= 2
        assert (row['stable_positive_point_evidence'] == 'True') == checks[row['contrast']]
        for field, source in [('both_nominal_lowers_positive', ci), ('both_family_lowers_positive', family)]:
            supported = all(float(next(r for r in source if r['contrast'] == row['contrast'] and r['metric'] == m)['low']) > 0 for m in PRIMARY_METRICS)
            assert (row[field] == 'True') == supported
    gate = read('QUALIFICATION_GATE')
    parent_gate = json.loads((parent / 'results/QUALIFICATION_GATE.json').read_text(encoding='utf-8'))
    parent_ok = parent_gate['qualified_for_separate_core_module_study']
    core_ok = all(checks[name] for name in ['EARLY_FILM', 'LATE_GATE', 'PRE_REFERENCE'])
    assert gate['parent_fixed_rule_pass'] == parent_ok and gate['all_three_retained_core_point_rules_pass'] == core_ok
    assert gate['correct_early_context_diagnostic_pass'] == checks['CORRECT_EARLY_CONTEXT']
    assert gate['final_study_planning_eligible'] == bool(parent_ok and core_ok)
    assert gate['status'] == ('PASS' if parent_ok and core_ok else 'FAIL')
    assert not gate['new_Outer_authorized'] and not gate['active_goal_achieved']
    complete = read('COMPLETION_STATUS')
    assert complete['status'] == 'CORE_MODULE_INTERNAL_COMPLETE_VERIFIED' and complete['fresh_fits'] == 75 and complete['exact_reused_fits'] == 15
    assert not complete['new_Outer'] and not complete['active_goal_achieved']
    assert len(table('POOLED_RESULTS')) == 18
    print(json.dumps(dict(status='PASS', public_package_members=len(allowed) + 1, scientific_sources=12, parent_sources=13,
                          arms=6, seeds=3, nominal_intervals=108, additional_intervals=8, private_data_used=False,
                          full_patient_bootstrap_recomputed=False, private_independent_verifier_required=True)))


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results')
