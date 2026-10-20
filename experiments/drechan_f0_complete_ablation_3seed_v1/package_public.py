"""Write actual verified results; export anonymous aggregates only."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(r'D:\drechan_f0_complete_ablation_3seed_v1_runtime')
PUB = ROOT/'public'
ALLOW = ['ADMISSION.json', 'ROLE_AUDIT.csv', 'PREPROCESSING_ADMISSION.csv', 'ADMISSION_REUSE.csv',
         'SOURCE_PUSH_BEFORE_TRAIN.json', 'TRAIN_STATUS.json', 'TRAIN_CELLS.csv', 'TRAIN_HISTORY.csv',
         'INTERNAL_VERIFICATION.json', 'VALIDATION_REPLAY_CHECKS.csv', 'PREPROCESSING_REPLAY.csv',
         'INITIALIZATION_AUDIT.csv', 'FREEZE_STATUS.json', 'PREDICTION_STATUS.json',
         'RAW_COVERAGE.csv', 'MAIN_RESULTS.csv', 'SEED_RESULTS.csv', 'SEED_SD.csv', 'SOURCE_SEED_RESULTS.csv', 'POOLED_SEED_MEAN.csv', 'FOLD_SEED_DIRECTIONS.csv', 'FOLD_RESULTS.csv', 'FOLD_SD.csv', 'SOURCE_RESULTS.csv',
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
    check=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    admission=json.loads((PUB/'ADMISSION.json').read_text(encoding='utf-8'))
    assert check['status']=='PASS' and check['patient_rows']==4560 and check['patient_metric_checks']==41040
    points,sd=table('MAIN_RESULTS'),table('SEED_SD')
    assert len(points)==19 and len(table('INTERVALS'))==351 and len(table('PRIMARY_FAMILY_INTERVALS'))==36
    lines=['# Fixed six-feature DREChannNet: complete three-seed ablation','',
        'All results are exploratory repeated-test, not independent confirmation. Current Full was fixed before these runs; B0 is a historical architecture reference only. No winning ablation is promoted. All285 selection cells froze before unfinished Outer inference, and all285 score cells sealed before this report.','',
        '19configurations × seeds42/43/44 × original5folds. Original80patients,7635channels,256seizures and1471965valid Raw windows, complete-patient updates and original FP32/TF32-off optimizer/loss/budgets are retained. Variance and head interaction remain absent. No new extraction, feature/hyperparameter search, Development performance gate or post-Outer tuning. The earlier33-patient internal three-seed experiment is not included.','',
        f"Strict historical reuses: {admission['expected_reused_cells']}; fresh fits: {admission['expected_fresh_fits']}. Mandatory45current seed42 cells plus only independently admitted historical Raw-only cells. Reuses preserve checkpoint/preprocessing/initialization/validation/threshold/prediction hashes. Old other-architecture deletion experiments are not reused.",'',
        '## Equal-patient metrics, then equal-seed mean ± sample SD','',
        '| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in points:
        v=next(z for z in sd if z['arm']==r['arm'])
        lines.append('| '+r['arm']+' | '+' | '.join(f"{float(r[m]):.6f} ± {float(v[m]):.6f}" for m in METRICS)+' |')
    lines+=['','The primary summaries average metrics, not probabilities, and do not select a best seed. Macro-F1 first averages EZ/NEZ F1 within a patient, then patients equally. EZ-fraction bias is predicted minus observed EZ fraction. Pooled-channel values are separately labeled per seed and their metric means; MRR and Top-1 remain patient-local.','',
        '[Individual seeds](SEED_RESULTS.csv) · [seed sample SD](SEED_SD.csv) · [folds](FOLD_RESULTS.csv) · [fold sample SD](FOLD_SD.csv) · [sources](SOURCE_RESULTS.csv) · [source-by-seed](SOURCE_SEED_RESULTS.csv) · [pooled](POOLED_RESULTS.csv) · [coverage](RAW_COVERAGE.csv)','',
        '## Paired evidence and direction consistency','',
        'Each patient metric is first averaged over seeds42/43/44.10000seed42 draws resample the same80unique patients.351nominal95% percentile intervals cover19absolute results,18Full-minus controls and2factorial interactions, each across9metrics.36additional Bonferroni-percentile intervals cover18AUROC/Macro-F1 contrasts (two-sided tail0.05/72). Finite bootstrap-tail resolution and historical repeated testing limit interpretation; these intervals are conditional on the fitted models, not population-level estimates of training-seed uncertainty.','',
        '| Contrast | ΔMacro-F1 | ΔAUROC | Seeds both positive | Fold/seed cells both positive | Additional-family lower bounds (F1,AUC) |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in table('COMPONENT_EVIDENCE'):
        lines.append(f"| {r['contrast']} | {float(r['macro_f1_difference']):+.6f} | {float(r['auroc_difference']):+.6f} | {r['seeds_both_positive']}/3 | {r['fold_seed_cells_both_positive']}/15 | {float(r['macro_family_low']):+.6f}, {float(r['auroc_family_low']):+.6f} |")
    lines+=['','[All nominal intervals](INTERVALS.csv) · [36additional intervals](PRIMARY_FAMILY_INTERVALS.csv) · [every fold/seed contrast direction](FOLD_SEED_DIRECTIONS.csv)','',
        '## Interpretation boundaries','',
        '- Feature PR deletion removes physiological-summary and physiological-window PR; Raw PR deletion removes final Raw96D patient Z only. BOTH removes all those relative operations but retains waveform normalization and self pre-ictal subtraction. Window-only deletion preserves corrected summary PR. No control replaces legitimate inputs with arbitrary zeros except declared feature/branch removals.',
        '- Both Guidance removes only Early FiLM and Late Gate, retaining direct physiological summary PR. Module removals change capacity; those contrasts do not by themselves separate information from optimization/capacity effects.',
        '- Each feature deletion masks both summary coordinates and its early/late window coordinate before relative statistics. Other coordinates and shared initial tensors remain unchanged; every new cell starts from matched complete initialization, not trained Full weights.',
        '- Early FiLM mismatch permutes only its already computed input within patient/seizure/exact full-mask groups, salt42 across all training seeds. Late/direct inputs stay correct. Singletons remain unchanged; the full support and actual changed/eligible/singleton window counts are reported. This is a context control, not proof of physiology or seizure propagation.',
        '- Raw-only reads no physiological features; Physiology-only reads no Raw. Their64-wide heads use the original corresponding input-weight slice and scaling, without width compensation.',
        '- PR interaction is Full − NoFeaturePR − NoRawPR + NoBothPR; guidance interaction is Full − NoEarly − NoLate + NoBothGuidance, computed per patient metric before averaging. They are statistical performance interactions, not biological interactions.',
        '- Three seeds assess limited training variability; folds overlap in training patients. Neither three seeds nor paired intervals turn a historically reused Outer cohort into independent confirmation. Single deletions do not establish global optimality of the six-feature set.',
        '- Small-source cells under five patients suppress source metric values only; all original patients/channels remain in the main results. No outcome-directed exclusions.',
        f"- Independent verification: {check['patient_rows']}patient/seed rows, {check['patient_metric_checks']}metric values,351nominal and36additional interval checks; maximum interval discrepancy {check['maximum_interval_difference']:.3g}.",
        '- Public release includes source/hashes/anonymous aggregates only. Signals, identities, patient/channel predictions, checkpoints, preprocessors, caches and runtime logs stay on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    status = json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))
    assert status['status'] == 'COMPLETE_VERIFIED_CFA3_19_GROUP_3SEED_FIVE_FOLD_OUTER'
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
