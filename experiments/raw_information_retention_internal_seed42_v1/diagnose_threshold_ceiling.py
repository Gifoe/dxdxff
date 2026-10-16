"""Internal-only retrospective oracle diagnostic, never a usable classifier.

Added during fitting, after some racing decisions were known. This is not an
original preregistered endpoint or evidence of attainable held-out performance.
The diagnostic never changes checkpoints, thresholds, racing, or any score.
"""
from scipy.special import expit
from common import *
from summarize import cluster_interval
import export_completed_results as exporter


def oracle_f1(y, p):
    order = np.argsort(-p, kind='stable')
    q, truth = p[order], y[order]
    # A score threshold includes whole tied groups. Label-aware tie-breaking
    # would create a fictitious upper bound for a different decision class.
    eligible = np.r_[q[:-1] != q[1:], True]
    k = np.arange(1, len(q) + 1)
    f1 = 2 * np.cumsum(truth) / (k + truth.sum())
    return float(max(0., f1[eligible].max()))


def main():
    binding = setup()
    terminal = jr(PUB / 'INTERNAL_TERMINAL.json')
    verification = jr(PUB / 'INDEPENDENT_VERIFICATION.json')
    assert terminal['status'] == 'COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'
    assert verification['status'] == 'PASS' and verification['input_binding_sha256'] == digest(binding)
    banks = legal_fit_banks(labels=True)
    paths = role_paths(banks)
    seal = jr(ROOT / 'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    assert seal['input'] == binding
    rows = []
    for r in seal['cells']:
        if r['screen'] != 1:
            continue
        fold, arm = r['fold'], r['arm']
        hp = ROOT / 'models_private' / f'fold{fold}_screen1' / arm / 'HELD_PREDICTIONS_PRIVATE.pkl'
        assert sha(hp) == r['held_predictions_sha256']
        saved = read(hp)
        d, held = banks[fold - 1], paths[(fold, 1)]['roles']['held']
        assert np.array_equal(saved['archive_indices'], d['archive_indices'][held])
        y, p = 1 - d['y'][held], expit(-saved['logit_nez'].astype(float))
        for pid in sorted(set(d['patient'][held])):
            loc = d['patient'][held] == pid
            prediction = p[loc] >= saved['threshold']
            actual = 2 * np.sum(prediction * y[loc]) / (prediction.sum() + y[loc].sum())
            oracle = oracle_f1(y[loc], p[loc])
            assert oracle >= actual - 1e-12
            rows.append(dict(arm=arm, patient=pid, actual_f1=actual,
                             oracle_f1=oracle, oracle_minus_actual=oracle - actual))
    frame = pd.DataFrame(rows)
    output = []
    for arm, g in frame.groupby('arm'):
        for r in cluster_interval(g.to_dict('records'), ['actual_f1', 'oracle_f1', 'oracle_minus_actual']):
            output.append(dict(arm=arm, **r))
    pd.DataFrame(output).to_csv(PUB / 'INTERNAL_ORACLE_THRESHOLD_DIAGNOSTIC.csv', index=False)
    write(PUB / 'ORACLE_DIAGNOSTIC_AUDIT.json', dict(
        status='PASS', input_binding_sha256=digest(binding),
        diagnostic_type='retrospective internal per-patient label-aware threshold ceiling',
        not_a_classifier=True, not_an_original_preregistered_endpoint=True,
        all_equal_score_ties_retained=True, all_nine_first_path_arms_retained=True,
        new_training_cells=0, new_score_inferences=0, new_Outer_passes=0,
        scientific_selection_and_threshold_rules_unchanged=True,
        interpretation='An optimistic finite-sample upper bound for threshold-only changes to these fixed scores, '
                       'not a prediction of achievable test performance or a legal deployable threshold.',
        private_rows_exported=False))
    exporter.FILES.extend(['RACED_PATH_PAIRED_INTERVALS.csv', 'RACED_PATH_SUMMARY_AUDIT.json',
                          'NORMALIZATION_ENCODER_INTERACTION.csv', 'REPORTING_FINALIZATION_AUDIT.json',
                          'INTERNAL_ORACLE_THRESHOLD_DIAGNOSTIC.csv', 'ORACLE_DIAGNOSTIC_AUDIT.json'])
    exporter.main(PUB)
    print('RETENTION_INTERNAL_ORACLE_DIAGNOSTIC_COMPLETE_NOT_A_CLASSIFIER', flush=True)


if __name__ == '__main__':
    main()
