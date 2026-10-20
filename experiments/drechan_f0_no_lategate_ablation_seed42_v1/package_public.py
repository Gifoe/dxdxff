"""Verified no-Late seed42 outcomes and explicit anonymous-only package."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1_runtime')
PUB=ROOT/'public'
ALLOW=['ADMISSION.json','ALIASES.json','ROLE_AUDIT.csv','PREPROCESSING_ADMISSION.csv','ADMISSION_REUSE.csv','INITIALIZATION_AUDIT.csv',
    'SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','INTERNAL_VERIFICATION.json',
    'VALIDATION_REPLAY_CHECKS.csv','PREPROCESSING_REPLAY.csv','FREEZE_STATUS.json','PREDICTION_STATUS.json','RAW_COVERAGE.csv',
    'MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv',
    'PRIMARY_FAMILY_INTERVALS.csv','COMPONENT_EVIDENCE.csv','FOLD_DIRECTIONS.csv','GOAL_AUDIT.json','POOLED_RESULTS.csv',
    'ALIAS_RESULTS.csv','REPORT_STATUS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','REPORT.md']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def report():
    verify=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert verify['status']=='PASS' and verify['patient_rows']==1440 and verify['patient_metric_checks']==12960
    points=table('MAIN_RESULTS');assert len(points)==18
    lines=['# Fixed no-Late-Gate DREChannNet: seed42 ablations','',
        'The main is the exact historical seed42 w/o Late Gate, selected following already viewed Outer results. This study is exploratory repeated-test, not independent confirmation. Variance and Head Interaction remain absent. No result in this run is used to redesign or reselect the main.','',
        'All16 current-architecture groups have Late Gate physically removed before training. PREVIOUS_FULL_WITH_LATE and B0_ORIGINAL are clearly labelled historical references and retain their old Gate only for comparison.18unique groups ×5folds =90cells:30 strict reuses,60 fresh fits. Original80patients/7635channels/256seizures/1471965validRaw250windows, seed42 and roles/optimizer/loss/FP32/TF32off/chunk4096/fullpatient/budgets are unchanged. No other seeds,33patient surrogate, extraction, hyperparameter search or performance gate.','',
        '## Equal-patient averages','',
        '| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in points:lines.append('| '+r['arm']+' | '+' | '.join(f"{float(r[m]):.6f}" for m in METRICS)+' |')
    lines+=['','Macro-F1 is the equal-patient average of within-patient EZ/NEZ F1. Pooled-channel metrics are separate. MRR/Top-1 use within-patient ranking with original channel-name ties. EZ-fraction bias is predicted minus observed EZ fraction: closeness to zero, not greater value, is desirable. [Folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).','',
        '## Aliases, not additional experiments','',
        'w/o Late Gate is the main itself. With Late fixed absent, w/o Both Guidance is exactly w/o Early FiLM, retaining direct FeaturePR. They are reported in ALIAS_RESULTS.csv but not double-counted in cells, rows or intervals. No Early/Late four-cell interaction is estimated from a matrix in which Late is fixed absent.','',
        '## Full-minus-control and paired uncertainty','',
        '10000seed42 draws resample80unique patients.18absolute results,17Full-minus controls and1PRinteraction across9metrics give324nominal95% percentile intervals.34additional Bonferroni-percentile intervals cover17Macro-F1/AUROC pairs (tail0.05/68). These approximate intervals are conditional on this seed and historical model choices, not training-seed robustness or independent confirmation. All favorable/unfavorable directions are retained.','',
        '| Contrast | Delta Macro-F1 | Delta AUROC | Folds both positive | Additional-family lower bounds (F1,AUC) |',
        '| --- | ---: | ---: | ---: | --- |']
    for r in table('COMPONENT_EVIDENCE'):
        lines.append(f"| {r['contrast']} | {float(r['macro_f1_difference']):+.6f} | {float(r['auroc_difference']):+.6f} | {r['folds_both_positive']}/5 | {float(r['macro_family_low']):+.6f}, {float(r['auroc_family_low']):+.6f} |")
    lines+=['','[All intervals](INTERVALS.csv) · [34additional-family intervals](PRIMARY_FAMILY_INTERVALS.csv) · [every fold direction](FOLD_DIRECTIONS.csv)','',
        '## Operations and interpretation boundaries','',
        '- w/o Feature PR removes physiological summary and window-relative PR; w/o Raw PR removes only final96DpatientZ; w/o BOTH removes both, retaining waveform normalization/self pre-reference. Window-only deletion preserves corrected summary PR.',
        '- Each feature deletion zeros both summary coordinates and its EarlyFiLM window coordinate before relative statistics, keeping width and other shared initialization. Late does not exist in new groups. All new fits initialize from complete untrained matched sources, not trained main weights.',
        '- Early FiLM mismatch routes only its correct computed conditioning input within patient/seizure/exact whole-mask support, salt42; direct physiology remains correctly aligned. Singleton groups remain unchanged and coverage is reported. This tests context dependence, not physiological causality.',
        '- Raw-only reads no explicit physiology; Physiology-only reads no Raw. Both use original64-head weight slices/scales without width compensation.',
        '- Module/branch removals change capacity. Raw may compensate omitted explicit physiology; single deletions cannot prove global feature optimality or irreducibility.',
        '- Full minus PREVIOUS_FULL_WITH_LATE isolates the fixed Gate removal in this inherited protocol. B0 also differs in summary preprocessing/window relative guidance, and is not solely a Late comparison.',
        '- PR interaction is Full minus NoFeaturePR minus NoRawPR plus NoBothPR on patient metrics. No biological interaction follows from a performance interaction.',
        '- All90models/preprocessors/epochs/thresholds froze before unfinished Outer; all90score cells sealed before metrics. Thirty reuses passed input/initialization/preprocessing/validation/epoch/threshold/checkpoint/score identity. Other with-Late PR/feature ablations were not reused.',
        '- Small-source metric cells below5patients are suppressed, not removed from the main cohort. Five folds overlap in training patients; their SD and direction counts are descriptive. No one-seed result establishes multi-seed robustness.',
        f"- Independent audit: {verify['patient_rows']}patient rows, {verify['patient_metric_checks']}metric values,324nominal/34additional intervals; maximum interval discrepancy {verify['maximum_interval_difference']:.3g}.",
        '- Public artifacts contain only source/hashes/anonymous aggregates. Signals, identities, individual scores, weights, preprocessors, caches and runtime logs remain on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    status = json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))
    assert status['status'] == 'COMPLETE_VERIFIED_NLG_18_GROUP_SEED42_FIVE_FOLD_OUTER'
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
