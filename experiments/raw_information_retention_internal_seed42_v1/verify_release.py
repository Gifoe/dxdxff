"""Independent public-artifact consistency checks, no private data dependencies."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from make_report import LABELS, METRICS, load_complete


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    root = here / 'results' / 'deterministic_execution_v2'
    terminal, verification, values = load_complete(root)
    read = lambda n: json.loads((root / n).read_text(encoding='utf-8'))
    manifest = read('PUBLIC_EXPORT_MANIFEST.json')
    assert manifest['status'] == 'PASS' and len(manifest['files']) == 21
    assert len({r['file'] for r in manifest['files']}) == 21
    for entry in manifest['files']:
        p = root / entry['file']
        assert p.parent == root and p.suffix in ['.csv', '.json'] and 'PRIVATE' not in p.name
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    source_gate = read('EXECUTION_SOURCE_ADMISSION.json')
    for name, expected in source_gate['source'].items():
        assert sha(here / name) == expected, ('Frozen scientific source modified', name)
    cells = pd.read_csv(root / 'INTERNAL_CELL_RESULTS.csv')
    assert not cells.duplicated(['fold', 'screen', 'arm']).any()
    assert len(cells) == manifest['executed_cells'] == verification['cells'] == terminal['executed_cells']
    assert int(cells.epochs.sum()) == terminal['formal_epochs']
    assert int(cells.patients.sum()) == verification['patient_rows']
    first = cells[cells.screen == 1]
    assert len(first) == 45 and set(first.arm) == set(LABELS)
    assert first.groupby('arm').patients.sum().eq(verification['first_path_logical_patients_per_arm']).all()
    maximum = 0.
    for arm, g in first.groupby('arm'):
        for metric in METRICS:
            error = abs(np.average(g[metric], weights=g.patients) - values.loc[arm, metric])
            assert error < 1e-12
            maximum = max(maximum, error)
    assert np.isfinite(values[list(METRICS)].to_numpy()).all()
    assert ((values[[m for m in METRICS if m != 'ez_fraction_bias']] >= 0) &
            (values[[m for m in METRICS if m != 'ez_fraction_bias']] <= 1)).all().all()
    races = read('RACING_DECISIONS.json')['decisions']
    assert len(cells) == 45 + 4 * sum(r['advanced'] for r in races)
    figure_qa = json.loads((here / 'figures' / 'FIGURE_QA.json').read_text(encoding='utf-8'))
    assert figure_qa['status'] == 'PASS_SOURCE_AND_VISUAL_REVIEW'
    widths = []
    for name in ['internal_information_retention', 'component_retention_effects']:
        svg = ET.parse(here / 'figures' / f'{name}.svg').getroot()
        width = float(svg.attrib['width'].removesuffix('pt')) * 25.4 / 72
        assert abs(width - 183) < 1e-6
        widths.append(width)
        for suffix in ['.svg', '.pdf', '.png', '.tiff']:
            assert (here / 'figures' / (name + suffix)).stat().st_size > 1000
    result = dict(status='PASS', anonymous_public_artifacts_verified=21,
                  frozen_v2_scientific_source_files_unchanged=len(source_gate['source']),
                  executed_cells=len(cells), executed_epochs=int(cells.epochs.sum()),
                  reconstructed_private_patient_rows=verification['patient_rows'],
                  first_path_appearances_per_arm=verification['first_path_logical_patients_per_arm'],
                  first_path_unique_patients=verification['first_path_unique_patients'],
                  main_vs_weighted_fold_max_error=maximum,
                  cal_score_replay_max_error=verification['calibration_replay_max_error'],
                  figure_widths_mm=widths, visual_review='PASS',
                  new_Outer_passes=0, independent_confirmation_claim=False,
                  goal_EZ_F1_greater_than_065_established=False,
                  raw_weights_scores_identities_runtime_logs_exported=False,
                  retrospective_oracle_diagnostic_not_a_classifier=True)
    (here / 'RELEASE_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
