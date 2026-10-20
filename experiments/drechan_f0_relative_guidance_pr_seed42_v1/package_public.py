"""Write actual verified results; export anonymous aggregates only."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(r'D:\drechan_f0_relative_guidance_pr_seed42_v1_runtime')
PUB = ROOT/'public'
ALLOW = ['ADMISSION.json', 'ROLE_AUDIT.csv', 'PREPROCESSING_ADMISSION.csv', 'ADMISSION_REUSE.csv',
         'SOURCE_PUSH_BEFORE_TRAIN.json', 'TRAIN_STATUS.json', 'TRAIN_CELLS.csv', 'TRAIN_HISTORY.csv',
         'INTERNAL_VERIFICATION.json', 'VALIDATION_REPLAY_CHECKS.csv', 'PREPROCESSING_REPLAY.csv',
         'NULL_CONTROL_RETRAIN_PARITY.csv', 'FREEZE_STATUS.json', 'PREDICTION_STATUS.json',
         'RAW_COVERAGE.csv', 'MAIN_RESULTS.csv', 'FOLD_RESULTS.csv', 'FOLD_SD.csv', 'SOURCE_RESULTS.csv',
         'METRIC_COVERAGE.csv', 'INTERVALS.csv', 'PRIMARY_FAMILY_INTERVALS.csv', 'COMPONENT_EVIDENCE.csv',
         'GOAL_AUDIT.json', 'POOLED_RESULTS.csv', 'REPORT_STATUS.json', 'INDEPENDENT_OUTER_VERIFICATION.json',
         'OUTER_STATUS.json', 'REPORT.md']
METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1', 'ez_fraction_bias']
NAMES = {'B0_ORIGINAL': 'Original 0.653828 main (exact reuse)', 'FULL': 'Fixed relative-guidance F0',
         'WO_FEATURE_PR': 'w/o physiological PR (summary and window)', 'WO_RAW_PR': 'w/o Raw PR',
         'WO_BOTH_PR': 'w/o BOTH PR', 'WO_WINDOW_PR': 'w/o relative window guidance',
         'WO_EARLY_FILM': 'w/o early FiLM', 'WO_LATE_GATE': 'w/o late Gate',
         'WO_PRE_REFERENCE': 'w/o own pre-ictal reference'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def report():
    check = json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    goal = json.loads((PUB/'GOAL_AUDIT.json').read_text(encoding='utf-8'))
    assert check['status'] == 'PASS'
    points, ci, family = table('MAIN_RESULTS'), table('INTERVALS'), table('PRIMARY_FAMILY_INTERVALS')
    components = table('COMPONENT_EVIDENCE')
    assert len(points) == 9 and len(ci) == 162 and len(family) == 14 and len(components) == 7
    lines = ['# Relative-guidance F0: original80-patient protocol', '',
        'This study modifies the exact historical six-feature no-Variance/no-head-interaction F0 architecture. The original CNN, pre-reference, stage/seizure pooling, late Gate and single64-hidden head remain. Full and all PR controls have11425parameters; deleting early FiLM or late Gate gives10377. No trained original weights initialize a fresh fit.', '',
        'All45models/preprocessors/validation-selected epochs and thresholds froze before new Outer inference; all45score cells sealed before metrics.40fresh fits and5exact original reference reuses. Original80patients,7635channels,256seizures and1471965valid Raw windows, original five patient-held-out roles, seed42, optimizer/loss/budget and threshold rule are unchanged. No extra Development admission performance gate is used.', '',
        'All results are exploratory repeated-test: the reference architecture and this historical Outer cohort have repeatedly been viewed. This is not independent confirmation. No configuration, epoch or threshold is selected from these Outer results.', '',
        '## Equal-patient averages', '',
        '| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in points:
        lines.append('| '+NAMES[r['arm']]+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines += ['', 'Macro-F1 averages EZ/NEZ F1 within each patient and then equally over80patients. Pooled-channel results are separate; ranking uses original channel-name tie breaking. EZ-fraction bias is predicted minus observed EZ fraction. [Folds](FOLD_RESULTS.csv), [sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).', '',
        '## Does PR improve both ranking and classification?', '',
        'The fixed contrasts are Full minus the stated control. Positive differences favor the candidate. All contrasts are reported, including unfavorable results.10000seed42same-patient bootstrap draws give162nominal95% intervals and14additional Bonferroni-level percentile intervals (seven contrasts times AUROC/Macro-F1, tail0.05/28). These are approximate exploratory intervals conditional on trained models; they do not correct historical architecture adaptation or measure training-seed uncertainty.', '',
        '| Contrast | Macro-F1 difference | AUROC difference | Folds both positive | Additional-family lows (F1, AUC) | Stable within this seed/protocol |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for r in components:
        lines.append(f"| {r['contrast']} | {float(r['macro_f1_difference']):+.6f} | {float(r['auroc_difference']):+.6f} | {r['folds_both_positive']}/5 | {float(r['macro_family_low']):+.6f}, {float(r['auroc_family_low']):+.6f} | {r['stable_within_this_seed_protocol']} |")
    lines += ['', 'The predeclared within-protocol evidence rule requires positive means for both metrics, both additional-family lower bounds above zero, and both metric directions positive in at least4of5folds. Fold counts are descriptive correlated estimates, not five independent replications.', '',
        '## Fixed objective audit', '', f"Overall fixed audit: **{goal['status']}**. No result was guaranteed and no winning control is promoted to a new main model.", '']
    for name, passed in goal['conditions'].items():
        lines.append(f'- {name}: {passed}')
    lines += ['', '## Interpretation and implementation boundaries', '',
        '- Summary Feature PR uses a FIT-unit-consistent degeneracy rule; relative window guidance blends the original FIT-global DELTA with complete-patient same-seizure/same-valid-time standardized channel contrast. The0.5mixture, epsilon and unsupported-window fallback were frozen before training. No new feature extraction, trainable capacity or Raw dual-view expansion is introduced.',
        '- w/o Feature PR removes BOTH summary patient normalization and relative physiological window guidance, restoring the original FIT-global inputs, not zeros. w/o Raw PR removes only the final Raw96patient normalization. w/o BOTH PR removes both; ordinary waveform normalization and own pre-reference remain. The five newly trained no-PR controls must exactly reproduce the original identical no-PR selected weights, validation, epochs and thresholds.',
        '- w/o window PR isolates the new window context from the direct-summary preprocessing repair. Thus Full minus original B0 is a joint preprocessing/context revision, not solely a window-context effect.',
        '- Early FiLM and late Gate removals reduce trainable capacity; their differences cannot isolate information content from capacity/optimization. No purely predictive association establishes physiology or seizure-propagation causality.',
        '- The PR interaction in INTERVALS.csv is Full minus NoFeaturePR minus NoRawPR plus NoBothPR, on each patient metric. It is a factorial performance interaction, not evidence of biological interaction.',
        '- Invalid/padded windows never enter channel statistics; valid unsupported contexts retain the original global input. No patient, channel, seizure or window is discarded for frequency quality or metric performance.',
        '- This one-seed experiment cannot establish seed robustness or independent external generalization. Source summaries suppress metric cells below five patients while retaining all patients in the main averages.',
        f"- Independent checks: {check['patient_rows']}patient rows, {check['patient_metric_checks']}metric values,162nominal and14additional intervals; maximum independent interval discrepancy {check['maximum_interval_difference']:.3g}.",
        '- Only source, hashes, anonymous counts and aggregate outcomes are public. Signals, identities, individual predictions, weights, preprocessors, caches and runtime logs remain on the trusted server.', '']
    (PUB/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    status = json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))
    assert status['status'] == 'COMPLETE_VERIFIED_RGF0_NINE_GROUP_FIVE_FOLD_OUTER'
    report()
    forbidden = {'patient', 'patient_id', 'channel', 'channel_id', 'archive_indices',
                 'prediction', 'probability', 'score_nez', 'y', 'path'}
    for name in ALLOW:
        path = PUB/name
        assert path.is_file() and path.stat().st_size < 5000000
        if path.suffix == '.csv':
            with path.open(encoding='utf-8', newline='') as f:
                assert not set(next(csv.reader(f))) & forbidden
        assert not any(value in path.read_text(encoding='utf-8') for value in ['password=', 'PRIVATE KEY', 'fit_patient_ids'])
    manifest = dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',
                    files={n: sha(PUB/n) for n in ALLOW}, private_artifacts_exported=False)
    out = ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:
            assert json.loads(z.read('PACKAGE_MANIFEST.json')) == manifest
    else:
        with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in ALLOW:
                z.write(PUB/name, name)
            z.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(file=out.name, bytes=out.stat().st_size, sha256=sha(out), files=len(ALLOW)+1)), flush=True)


if __name__ == '__main__':
    main()
