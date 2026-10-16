"""Joint 2x2 interaction and raced-path uncertainty from sealed saved results."""
import summarize_raced_paths
from common import *
from summarize import cluster_interval
import export_completed_results as exporter


def main():
    summarize_raced_paths.main()
    verification = jr(PUB / 'INDEPENDENT_VERIFICATION.json')
    assert verification['status'] == 'PASS'
    frame = pd.DataFrame(read(ROOT / 'PATIENT_RESULTS_PRIVATE.pkl'))
    first = frame[frame.screen == 1]
    keys = ['fold', 'screen', 'patient', 'center']
    arms = ['C0_WINDOW_SHORT', 'C1_PRE_SHORT', 'C2_WINDOW_LONG', 'C3_PRE_LONG']
    joined = None
    for j, arm in enumerate(arms):
        g = first[first.arm == arm][keys + METRICS].rename(columns={m: f'{m}_{j}' for m in METRICS})
        joined = g if joined is None else joined.merge(g, on=keys, validate='one_to_one')
    assert len(joined) == verification['first_path_logical_patients_per_arm']
    differences = pd.DataFrame({'patient': joined.patient, **{
        m: joined[f'{m}_3'] - joined[f'{m}_2'] - joined[f'{m}_1'] + joined[f'{m}_0']
        for m in METRICS}})
    result = cluster_interval(differences.to_dict('records'))
    pd.DataFrame(result).to_csv(PUB / 'NORMALIZATION_ENCODER_INTERACTION.csv', index=False)
    write(PUB / 'REPORTING_FINALIZATION_AUDIT.json', dict(
        status='PASS', input_binding_sha256=verification['input_binding_sha256'],
        source_private_table_sha256=sha(ROOT / 'PATIENT_RESULTS_PRIVATE.pkl'),
        matched_first_path_appearances=len(joined), unique_patients=joined.patient.nunique(),
        interaction='(C3_PRE_LONG-C2_WINDOW_LONG)-(C1_PRE_SHORT-C0_WINDOW_SHORT)',
        uncertainty='joint four-arm patient-ID cluster bootstrap, not combined marginal intervals',
        attribution='encoder family, not isolated receptive-field change',
        executed_training_cells_unchanged=verification['cells'], new_score_inferences=0,
        new_Outer_passes=0, individual_rows_exported=False))
    exporter.FILES.extend(['NORMALIZATION_ENCODER_INTERACTION.csv', 'REPORTING_FINALIZATION_AUDIT.json'])
    exporter.main(PUB)
    print('RETENTION_REPORTING_FINALIZED_NO_NEW_PREDICTIONS', flush=True)


if __name__ == '__main__':
    main()
