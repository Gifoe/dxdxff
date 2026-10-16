"""Local public-artifact completion audit; never loads private data or models."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE / 'results' / 'final_stage_v1'
INNER = HERE / 'results' / 'internal_v1'
PROCEDURES = {'NATIVE', 'FROZEN_PRE', 'WINDOW_MEAN', 'WINDOW_LOGENERGY',
              'PRE_MEAN', 'PRE_LOGENERGY', 'SELECTED'}
METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy',
           'mrr', 'hit_at_1', 'ez_fraction_bias']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    training = read_json(FINAL / 'TRAINING_STATUS.json')
    outer = read_json(FINAL / 'OUTER_STATUS.json')
    terminal = read_json(FINAL / 'FINAL_TERMINAL.json')
    verification = read_json(FINAL / 'INDEPENDENT_FINAL_VERIFICATION.json')
    statistics = read_json(FINAL / 'INDEPENDENT_STATISTICS_VERIFICATION.json')
    support = read_json(FINAL / 'RAW_SUPPORT_AUDIT.json')
    assert training['status'] == 'ALL35_FRESH_FINAL_MODELS_FROZEN'
    assert training['actual_fresh_fits'] == 20 and training['logical_cells'] == 35
    assert outer['status'] == 'ALL35_FROZEN_FRESH_RAW_SCORE_CELLS_SEALED'
    assert terminal['status'] == 'COMPLETE_SEVEN_FRESH_RAW_PROCEDURES'
    assert terminal['primary'] == verification['primary'] == 'SELECTED'
    assert verification['status'] == statistics['status'] == support['status'] == 'PASS'
    assert verification['patient_metric_rows'] == 560
    assert verification['all9_patient_metrics_independent']
    assert verification['all_four_initializers_exact_per_fold']
    assert verification['source_role_and_archive_identity_exact']
    assert verification['selected_aliases_exact']
    assert verification['reference_predictions_unchanged']
    assert verification['maximum_metric_error'] < 1e-12
    assert statistics['independent_bootstrap_metric_checks'] == 162
    assert statistics['maximum_bootstrap_difference'] < 1e-12
    assert statistics['fold_means_and_sampleSD_exact']
    assert statistics['all9_metric_coverage_80perprocedure']
    assert statistics['small_source_suppression_verified']
    assert outer['score_seal_sha256'] == verification['score_seal_sha256'] == statistics['score_seal_sha256'] == support['score_seal_sha256']
    counts = support['counts']
    assert [counts[k] for k in ['patients', 'canonical_channels', 'seizure_records', 'valid_windows']] == [80, 7635, 256, 1471965]
    assert sum(counts[k] for k in ['pre_windows', 'phase_0to8_windows', 'phase_8to20_windows', 'at_or_after20_windows']) == counts['valid_windows']
    assert support['inference_valid_window_count_matches_independent_mask_census']
    assert support['no_training_or_prediction'] and support['no_masks_or_data_changed']
    selections = table(FINAL / 'FINAL_TRAINING_SELECTION.csv')
    assert len(selections) == 20 and all(int(r['selected_epoch']) >= 1 for r in selections)
    assert all(r['random_start'] == 'True' and int(r['parameters']) == 12529 for r in selections)
    means = {r['procedure']: r for r in table(FINAL / 'OUTER_PATIENT_MEAN.csv')}
    assert set(means) == PROCEDURES
    assert all(math.isfinite(float(r[m])) for r in means.values() for m in METRICS)
    assert all(abs(float(r['macro_f1']) - .5 * (float(r['f1']) + float(r['nez_f1']))) < 1e-12
               for r in means.values())
    assert abs(float(means['NATIVE']['macro_f1']) - .6304930529317453) < 1e-12
    assert len(table(FINAL / 'OUTER_FOLDS.csv')) == 35
    coverage = table(FINAL / 'OUTER_METRIC_COVERAGE.csv')
    assert len(coverage) == 63
    assert all(int(r['finite_patients']) == int(r['total_patients']) == 80 for r in coverage)
    choices = table(INNER / 'OWN_FIT_SELECTION.csv')
    assert [r['selected_arm'] for r in choices] == ['FROZEN_PRE', 'NATIVE', 'PRE_LOGENERGY', 'PRE_LOGENERGY', 'PRE_LOGENERGY']
    point = float(means['SELECTED']['macro_f1'])
    assert terminal['primary_point_over_065'] == (point > .65)
    assert not terminal['postOuter_tuning'] and not terminal['outer_oracle']
    assert terminal['scientific_status'] == 'EXPLORATORY_REPEATED_TEST'
    assert not terminal['independent_confirmation']
    result = dict(
        status='PASS', scope='public_delivery_consistency_not_a_new_private_replay',
        target_metric='80-patient equal-weight Macro-F1', primary='SELECTED',
        target_strictly_greater_than=.65, primary_value=point,
        primary_minus_native=point-float(means['NATIVE']['macro_f1']),
        target_point_achieved=point > .65,
        scientific_status='EXPLORATORY_REPEATED_TEST', independent_confirmation=False,
        no_secondary_winner_adoption=True, all_seven_complete_procedures_reported=True,
        fresh_random_start_fits=20, logical_model_cells=35, patient_metric_rows=560,
        all_nine_metrics_per_procedure_have_80_patients=True,
        independently_checked_CI_metric_rows=162,
        raw_support_counts=counts,
        late_windows_not_claimed_as_classifier_support=True,
        no_new_training_inference_or_threshold_optimization=True,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(FINAL.glob('*')) if p.is_file() and p.suffix in {'.csv', '.json'} and p.name != 'DELIVERY_COMPLETION_AUDIT.json'})
    destination = FINAL / 'DELIVERY_COMPLETION_AUDIT.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('PUBLIC_COMPLETION_AUDIT_PASS', 'primary Macro-F1:', point, 'target achieved:', point > .65)


if __name__ == '__main__':
    main()
