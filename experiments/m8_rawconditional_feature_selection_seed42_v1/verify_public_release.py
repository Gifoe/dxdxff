"""Verify transferred public bytes, protected v2 release and plotted aggregates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = HERE / 'results' / 'dense_reference'
    manifest = json.loads((result / 'PUBLIC_TRANSFER_MANIFEST.json').read_text())
    assert manifest['status'] == 'PASS' and len(manifest['files']) == 19
    for name, entry in manifest['files'].items():
        path = result / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    parent = HERE.parent / 'm8_physiology_discovery_selection_seed42_v2'
    release = json.loads((parent / 'RELEASE_MANIFEST.json').read_text())
    for name, entry in release['files'].items():
        path = parent / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    selected = pd.read_csv(result / 'FROZEN_INTERNAL_SELECTION.csv')
    assert selected[selected.recipe == 'SELECTED'].k.tolist() == [1, 0, 9, 2, 1]
    record = pd.read_csv(result / 'SEARCH_RECORD.csv')
    assert len(record) == 9179 and not any('patient' in col or 'channel' in col for col in record.columns)
    assert json.loads((result / 'INTERNAL_REPLAY_VERIFICATION.json').read_text())['status'] == 'PASS'
    controls = pd.read_csv(result / 'CHOICE_CONTROL_COMPARISON.csv')
    assert len(controls) == 30 and controls.selection_conditioned.all()
    figures = HERE / 'figures'
    for name in ['01_conditional_cardinality_stability', '02_internal_increment_source']:
        for ext in ['svg', 'pdf', 'png', 'tiff']:
            assert (figures / f'{name}.{ext}').stat().st_size > 5000
    main = pd.read_csv(result / 'INTERNAL_MAIN_RESULTS.csv').set_index('model')
    paired = pd.read_csv(result / 'INTERNAL_PAIRED_INTERVALS.csv')
    for row in paired.itertuples(index=False):
        assert abs(main.loc['SELECTED', row.metric] - main.loc[row.reference, row.metric] - row.delta) < 1e-12
    summary = pd.read_csv(figures / 'SOURCE_CARDINALITY.csv')
    assert len(summary) == 10 and (summary.patient_appearances == 255).all()
    assert np.isfinite(summary.select_dtypes('number')).all().all()
    # The manuscript tables are transcribed from the released full-precision
    # aggregates, not from an unrun model or the failed numerical path.
    table_metrics = ['auroc', 'auprc', 'macro_f1', 'f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1']
    aliases = {'Raw-only192D':'RAW_ONLY', 'Raw + old7':'OLD7', 'Raw + own-FIT-selected groups':'SELECTED'}
    checked_cells = 0
    for line in (HERE / 'REPORT.md').read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):
            continue
        values = [v.strip() for v in line.strip('|').split('|')]
        if values[0] in aliases:
            for metric, value in zip(table_metrics, values[1:]):
                assert abs(float(value) - main.loc[aliases[values[0]], metric]) <= 5.1e-7
                checked_cells += 1
    assert checked_cells == 24
    outer = pd.read_csv(parent / 'results' / 'outer' / 'MAIN_RESULTS.csv').set_index('model')
    aliases = {'D1 selected':'D1__SELECTED', 'D1 matched old7':'D1__OLD7',
               'M8 selected':'M8__SELECTED', 'M8 matched old7':'M8__OLD7',
               'Concat selected':'Concat__SELECTED', 'Concat matched old7':'Concat__OLD7',
               'Matched Raw-only0–20':'Concat__RAW_ONLY', 'Matched Raw-only0–30':'Concat__RAW_SUPPORT30',
               'Historical original M8':'HIST_M8'}
    synthesis = HERE.parent / 'm8_physiology_goal_evidence_audit_seed42_v1' / 'RESEARCH_SYNTHESIS.md'
    for line in synthesis.read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):
            continue
        values = [v.strip() for v in line.strip('|').split('|')]
        if values[0] in aliases:
            for metric, value in zip(table_metrics, values[1:]):
                assert abs(float(value) - outer.loc[aliases[values[0]], metric]) <= 5.1e-7
                checked_cells += 1
    assert checked_cells == 96
    status = dict(status='PASS', server_transfer_files=19, original_release_files_unchanged=len(release['files']),
        raw_encoder_fits=20, raw_encoder_epochs=286, accepted_reference_head_fits=9344,
        additional_registered_choice_control_fits=25, failed_block_head_fits_retained_not_accepted=9344,
        all_search_evaluations_replayed=9179, all_assessment_fits_replayed=165,
        independent_assessment_patient_metric_rows=2805, new_Outer_inference=False,
        report_metric_cells_checked_against_full_precision=checked_cells,
        stable_global_minimum_or_clinical_increment_proven=False,
        no_private_patient_channel_rows_or_predictions=True)
    (HERE / 'RELEASE_STATUS.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
