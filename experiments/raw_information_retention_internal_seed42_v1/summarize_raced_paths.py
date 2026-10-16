"""Reporting-only paired uncertainty for the already executed raced paths.

Requires the original independent terminal; no fitting or score inference.
"""
from common import *
from summarize import cluster_interval
import export_completed_results as exporter


def main():
    binding = setup()
    terminal = jr(PUB / 'INTERNAL_TERMINAL.json')
    verification = jr(PUB / 'INDEPENDENT_VERIFICATION.json')
    assert terminal['status'] == 'COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'
    assert verification['status'] == 'PASS' and verification['input_binding_sha256'] == digest(binding)
    seal = jr(ROOT / 'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    assert seal['input'] == binding
    frame = pd.DataFrame(read(ROOT / 'PATIENT_RESULTS_PRIVATE.pkl'))
    assert len(frame) == verification['patient_rows']
    joined_paths = []
    records = []
    for decision in seal['racing']:
        if not decision['advanced']:
            continue
        fold, arm = decision['fold'], decision['chosen']
        g = frame[(frame.fold == fold) & frame.screen.isin([2, 3])]
        left, right = g[g.arm == arm], g[g.arm == 'C0_WINDOW_SHORT']
        j = left.merge(right, on=['fold', 'screen', 'patient', 'center'],
                       suffixes=('_l', '_r'), validate='one_to_one')
        assert len(j) == len(left) == len(right) and set(j.screen) == {2, 3}
        differences = pd.DataFrame({'patient': j.patient, **{
            m: j[m + '_l'] - j[m + '_r'] for m in METRICS}})
        for r in cluster_interval(differences.to_dict('records')):
            records.append(dict(scope='one_original_FIT_additional_paths', fold=fold, left=arm,
                                right='C0_WINDOW_SHORT', **r))
        joined_paths.append(j)
    if joined_paths:
        pooled = pd.concat(joined_paths, ignore_index=True)
        differences = pd.DataFrame({'patient': pooled.patient, **{
            m: pooled[m + '_l'] - pooled[m + '_r'] for m in METRICS}})
        for r in cluster_interval(differences.to_dict('records')):
            records.append(dict(scope='advanced_FITs_only_additional_paths', fold=0,
                                left='registered_per_FIT_raced_candidate',
                                right='C0_WINDOW_SHORT', **r))
    columns = ['scope', 'fold', 'left', 'right', 'metric', 'mean', 'low', 'high',
               'logical_patient_appearances', 'unique_patients', 'draws', 'interval']
    pd.DataFrame(records, columns=columns).to_csv(PUB / 'RACED_PATH_PAIRED_INTERVALS.csv', index=False)
    write(PUB / 'RACED_PATH_SUMMARY_AUDIT.json', dict(
        status='PASS', input_binding_sha256=digest(binding),
        source_private_table_sha256=sha(ROOT / 'PATIENT_RESULTS_PRIVATE.pkl'),
        advanced_original_FITs=len(joined_paths), rows=len(records),
        executed_training_cells_unchanged=verification['cells'], new_score_inferences=0,
        independent_replication=False, first_path_selection_scores_excluded=True,
        repeated_patient_IDs_carried_together=True,
        interpretation='Descriptive paired uncertainty for registered additional paths; '
                       'overlapping patients/roles and selection condition prevent independent confirmation.',
        individual_rows_exported=False))
    exporter.FILES.extend(['RACED_PATH_PAIRED_INTERVALS.csv', 'RACED_PATH_SUMMARY_AUDIT.json'])
    exporter.main(PUB)
    print('RETENTION_RACED_UNCERTAINTY_COMPLETE', len(joined_paths), flush=True)


if __name__ == '__main__':
    main()
