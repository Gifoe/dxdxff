"""Report actual verified aggregates and export only a strict anonymous allowlist."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','INTERNAL_VERIFICATION.json','VALIDATION_REPLAY_CHECKS.csv',
    'FREEZE_STATUS.json','PREDICTION_STATUS.json','RAW_COVERAGE.csv','MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv',
    'MACRO_FAMILY_INTERVALS.csv','POOLED_RESULTS.csv','HISTORICAL_CLASSIFIER_REFERENCE.csv','HISTORICAL_CLASSIFIER_AUDIT.json','REPORT_STATUS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','REPORT.md']
NAMES={'FULL_F0':'Main: six features, no head interaction','WO_EARLY_FILM':'w/o early FiLM','WO_BOTH_GUIDANCE':'w/o both guidance','WO_BOTH_PR':'w/o BOTH PR',
    'WO_RAW_PRE_REFERENCE':'w/o Raw pre-reference','EARLY_FILM_MISMATCH':'Early FiLM mismatch','RAW_ONLY':'Raw-only','PHYSIOLOGY_ONLY':'Physiology-only',
    'DROP_BETA':'w/o beta','DROP_LOW_GAMMA':'w/o low gamma','DROP_PARTIAL_HIGH_GAMMA':'w/o partial high gamma80-125Hz','DROP_RMS':'w/o RMS','DROP_LINE_LENGTH':'w/o line length','DROP_SPECTRAL_ENTROPY':'w/o entropy10-125Hz'}
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def report():
    check=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'));assert check['status']=='PASS'
    points=table('MAIN_RESULTS');ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(points)==14 and len(ci)==243 and len(family)==13
    lines=['# Six-feature F0 without head interaction: approved fourteen-group ablation','',
        'The main model is the exact previously frozen six-feature WO_HEAD_INTERACTION configuration. Variance and head interaction remain absent in every group. Original80patients,7635channels,256seizures; seed42 and original five patient-held-out folds. All70selected models/preprocessors/thresholds froze before newOuter; all70score cells sealed before unified reporting.55genuine fresh fits and15exact reused cells.','',
        'This reference was chosen after historical Outer viewing. All results are exploratory repeated-test, not independent confirmation. No group, threshold or model was chosen or changed from this pass. No Development eligibility filter is used.','',
        '## Equal-patient averages','',
        '| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in points:lines.append('| '+NAMES[r['arm']]+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','Macro-F1 is the mean of EZ-F1 and NEZ-F1 for each patient, then averaged equally over80patients. Pooled-channel metrics are separate. EZ-fraction bias is predicted minus observed EZ fraction; zero is preferable. See [folds](FOLD_RESULTS.csv), [sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).','',
        '## Patient-paired differences','',
        'Full minus ablation; positive favors the fixed main model.10000seed42same-patient bootstrap draws.243nominal95% intervals cover14absolute and13paired contrasts across9metrics.13additional Macro-F1 intervals use Bonferroni-level percentile tails0.05/(2x13); approximate exploratory intervals, not exact simultaneous coverage or correction for historical researcher selection.','',
        '| Ablation | Macro-F1 difference | Nominal95% interval | Additional13-comparison interval |',
        '| --- | ---: | --- | --- |']
    for r in ci:
        if r['kind']!='paired' or r['metric']!='macro_f1':continue
        f=next(q for q in family if q['contrast']==r['contrast'])
        lines.append('| '+NAMES[r['contrast'].removeprefix('FULL_minus_')]+f" | {float(r['mean']):+.6f} | [{float(r['low']):.6f}, {float(r['high']):.6f}] | [{float(f['low']):.6f}, {float(f['high']):.6f}] |")
    lines+=['','## Interpretation limits','',
        '- The user excluded the isolated Late Gate deletion and interaction add-back. No new results for these rows are generated. A full early-by-late factorial interaction is not identifiable and is not reported.',
        '- w/o PR removes BOTH physiological-summary patient Z and final Raw96 patient Z. Per-window waveform normalization and Raw pre-reference remain unchanged.',
        '- Each single-feature deletion masks both summary coordinates and matching early/late window inputs during fresh training. Variance is always masked. Input widths and remaining shared initial tensors stay matched; one64-hidden classifier is retained.',
        '- Mismatch changes early FiLM donor trajectories only, within same patient/seizure/identical full support; direct physiology and late Gate remain correctly aligned. Singleton coverage is retained and reported.',
        '- Raw-only and Physiology-only retain the corresponding original head slices and branch scaling. Branch comparisons may include parameter-capacity/optimization differences, not only information content.',
        '- No single deletion proves global feature optimality; Raw may encode correlated physiological properties. Partial high gamma is80-125Hz, not full80-150Hz. Spectral entropy is10-125Hz.',
        '- Previously completed seven classifiers in both PR settings are separately reused as [historical context](HISTORICAL_CLASSIFIER_REFERENCE.csv); they are unaffected by neural head interaction. No fresh classical fit or classifier comparison is included in this14-group CI family.',
        '- One training seed and descriptive source summaries do not establish seed or external-cohort generalization. Patient bootstrap does not cover model-reference selection or training randomness.',
        f"- Independently verified {check['patient_rows']} patient rows, {check['patient_metric_checks']} metric values,243nominal intervals and13additional intervals. Maximum independent interval discrepancy {check['maximum_interval_difference']:.3g}.",
        '- Only code, hashes and anonymous aggregate reports are public. Signals, patient/channel identities, predictions, weights, preprocessors, mismatch maps and runtime logs remain on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    assert json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER'
    report();forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for n in ALLOW:
        p=PUB/n;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in ALLOW:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(ALLOW)+1)),flush=True)

if __name__=='__main__':main()
