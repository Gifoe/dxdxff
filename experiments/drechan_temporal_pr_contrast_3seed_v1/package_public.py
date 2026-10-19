"""Strict anonymous export and evidence-bound report; never export private cells."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(r'D:\drechan_temporal_pr_contrast_3seed_v1_runtime');PUB=ROOT/'public'
BASE=['ADMISSION.json','INTERNAL_ROLE_ADMISSION.csv','SOURCE_PUSH_BEFORE_TRAIN.json','REPORT.md','COMPLETION_STATUS.json']
TABLES=['TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','ROLE_AUDIT.csv','CALIBRATION_REPLAY.csv','SELECTION_VERIFICATION.json','FREEZE_STATUS.json',
    'PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv','SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv',
    'METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','POOLED_RESULTS.csv','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
NAMES={'B0':'Current six-feature single head','B1':'Temporal patient-relative contrast','B2':'Current TRAIN-only global reference','B3':'Wrong-time within-patient reference'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def table(stage,n):
    with (PUB/stage/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def report(stages,g):
    lines=['# Temporal patient-relative contrast: three-seed controlled experiment','',
        'All results are exploratory repeated-test on the historically viewed cohort. New within-FIT assessment roles prevent new epoch/threshold selection from using their own assessment labels; they do not erase historical architecture selection or create an independent cohort.','',
        '## Fixed design','',
        'B0 is the exact six-feature, no-Variance/no-explicit-head-interaction main model. B1/B2/B3 add the same 1776-parameter residual48-to24-to24 module; all share exact initial B0 outputs. Early FiLM, late Gate, losses, input scaling, original patient-held-out folds and budgets remain unchanged. Seeds42/43/44 are all reported.','',
        'B1 uses dimension-wise leave-one-channel-out median/MAD at the same valid seizure/time. B2 uses all current TRAIN windows at the same time, recomputed with current CNN weights before every optimizer update. B3 shifts only the reference to other post times with exactly the same full channel-support vector within the same seizure; singleton groups remain unchanged. All reference backgrounds are stop-gradient; recipient/fusion/pooling and final96D patient Z remain differentiable.','',
        'Internal roles are new TRAIN/CAL/ASSESS partitions entirely inside each original FIT. TRAIN alone determines preprocessing, scale and B2 reference; CAL alone selects epochs/thresholds. All60 selected cells freeze and replay before assessment; all60 score cells seal before endpoints. No original Development or ownOuter is used in this first stage.','',
        '## Predeclared continuation decision','',
        f"Performance margin gate: **{g['performance_margin_pass']}**. Current-TRAIN global-reference comparison gate: **{g['global_reference_comparison_pass']}**. Continue final stage: **{g['continue_final']}**.",
        'The1pp rule applies simultaneously to AUROC, AP, Macro-F1 and EZ-F1 mean differences B1-B0, with at least2/3seeds individually within all4margins. This is an engineering point gate, not formal statistical noninferiority. B1 must also exceed B2 on both ranking metrics on average and in at least2/3seeds, without more than1pp mean Macro/EZ loss. B3 is a diagnostic, not a selector. No gate was relaxed after results.','']
    if not g['continue_final']:lines+=['The fixed internal gates failed. No new final fits or Outer evaluation were authorized by this protocol. B1 is not adopted as a replacement. A failure here does not prove all forms of patient-relative learning ineffective.','']
    for stage in stages:
        check=jr(PUB/stage/'INDEPENDENT_VERIFICATION.json');assert check['status']=='PASS'
        lines+=['## '+stage.upper()+' equal-patient endpoints','',
            '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in table(stage,'MAIN_RESULTS'):lines.append('| '+NAMES[r['arm']]+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
        lines+=['','Primary points average patient-level metrics across all3seeds, not predictions or best seeds. Repeated internal assessment-fold appearances are averaged within each unique patient before equal-patient aggregation.','',
            '| Contrast | Endpoint | Difference | Nominal95% interval | Additional12-comparison interval |','| --- | --- | ---: | --- | --- |']
        family=table(stage,'PRIMARY_FAMILY_INTERVALS')
        for r in table(stage,'INTERVALS'):
            if r['kind']!='paired' or r['metric'] not in ['auroc','auprc','macro_f1','f1']:continue
            f=next(z for z in family if (z['contrast'],z['metric'])==(r['contrast'],r['metric']))
            lines.append(f"| {r['contrast']} | {r['metric']} | {float(r['mean']):+.6f} | [{float(r['low']):.6f}, {float(r['high']):.6f}] | [{float(f['low']):.6f}, {float(f['high']):.6f}] |")
        lines+=['',f"Independently verified {check['patient_fold_seed_rows']} patient-fold-seed rows and {check['patient_metric_checks']} endpoint values,63nominal and12additional interval values. Unique patients: {check['unique_patients']}.",
            f'See [{stage} seed results]({stage}/SEED_RESULTS.csv), [seed SD]({stage}/SEED_SD.csv), [folds]({stage}/FOLD_RESULTS.csv), [fold SD]({stage}/FOLD_SD.csv), [sources]({stage}/SOURCE_RESULTS.csv), [coverage]({stage}/COVERAGE.csv), [pooled endpoints]({stage}/POOLED_RESULTS.csv).','']
    lines+=['## Interpretation boundaries','',
        '- Patient AUROC/AP can change because contextual comparison occurs before nonlinear representation learning. Patient-wise monotone normalization of final logits alone cannot change within-patient rankings; that is not the method tested here.',
        '- B1 versus B0 measures incremental information plus additional capacity/optimization. B1 versus B2 matches added capacity/input width/scale recipe/compute updates; it contrasts local context with a global reference, not every possible non-PR architecture.',
        '- Correct-time evidence requires B1 versus B3 support and adequate changed-reference coverage. A positive point difference or mostly unchanged singleton control cannot establish timing specificity.',
        '- Both original summary-PR operations remain enabled in every arm. This experiment isolates additional temporal PR; it is not a claim that all PR is necessary or that deleting PR should break the network.',
        '- Historical PatientZ AUROC+0.028655 and[0.010792,0.049047] were factorial mean/min/max over8 frozen-CNN probe backgrounds, NOT a patient-paired95% confidence interval. Past weak relation/affine/ranking results remain negative evidence, not proof that this design must fail or succeed.',
        '-10000seed42unique-patient paired bootstrap draws use seed-averaged metrics; conditional intervals do not cover training randomness, overlapping FIT histories or prior researcher selection. Three seeds and source summaries do not establish external stability. Additional intervals use approximate Bonferroni percentile tails0.05/24; they are not exact simultaneous inference.',
        '- No physiological propagation or clinical mechanism is inferred from latent comparisons. Partial high gamma is80-125Hz, entropy10-125Hz; unavailable low-frequency sources are not invented.',
        '- No post-assessment/Outer architecture, threshold, cohort, gate, seed or loss tuning. No best-Outer arm is selected. No guarantee of improvement or global optimality.',
        '- Signals, identifiers, per-patient/channel predictions, weights, preprocessors, raw caches, global-reference tensors and runtime logs stay on the trusted server. Public outputs contain only source, hashes and anonymous aggregates.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

def main():
    assert jr(PUB/'internal/INDEPENDENT_VERIFICATION.json')['status']=='PASS';g=jr(PUB/'internal/GATE.json')
    stages=['internal']
    if g['continue_final']:
        assert jr(PUB/'final/INDEPENDENT_VERIFICATION.json')['status']=='PASS';stages+=['final']
    else:assert not (PUB/'final/TRAIN_STATUS.json').exists(),'No final stage permitted after failed gate'
    status=dict(status='COMPLETE_VERIFIED_FINAL_OUTER' if g['continue_final'] else 'COMPLETE_VERIFIED_INTERNAL_GATE_FAIL_NO_NEW_OUTER',
        stages=stages,internal_gate=g['status'],new_outer_completed=bool(g['continue_final']),scientific_status='EXPLORATORY_REPEATED_TEST',private_artifacts_exported=False)
    (PUB/'COMPLETION_STATUS.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    report(stages,g);allow=BASE+['internal/GATE.json']+[s+'/'+n for s in stages for n in TABLES]
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for n in allow:
        p=PUB/n;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in allow},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in allow:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(allow)+1)),flush=True)

if __name__=='__main__':main()
