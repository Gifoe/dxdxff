"""Explicit public allowlist, released only after all255scores and independent PASS."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','ROLE_AUDIT.csv','INITIALIZATION_AUDIT.csv','ADMISSION_REUSE.csv','POOLING_SUPPORT_ADMISSION.csv',
 'ALIAS_AUDIT.json','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
 'INTERNAL_VERIFICATION.json','VALIDATION_REPLAY_CHECKS.csv','PREPROCESSING_REPLAY.csv','FREEZE_STATUS.json',
 'PREDICTION_STATUS.json','RAW_COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv','SEED_SD.csv','FOLD_RESULTS.csv',
 'FOLD_SD.csv','SOURCE_RESULTS.csv','SOURCE_SEED_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv',
 'PRIMARY_FAMILY_INTERVALS.csv','COMPONENT_EVIDENCE.csv','FOLD_SEED_DIRECTIONS.csv','POOLED_RESULTS.csv',
 'POOLED_SEED_MEAN.csv','SEED_SELECTION_AUDIT.csv','GOAL_AUDIT.json','REPORT_STATUS.json',
 'INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','REPORT.md']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    status=json.loads((PUB/'OUTER_STATUS.json').read_text())
    assert status['status']=='COMPLETE_VERIFIED_NGA_17_GROUP_SELECTED3SEED_FIVE_FOLD_OUTER'
    verify=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text())
    assert verify['status']=='PASS' and verify['patient_rows']==4080 and verify['patient_metric_checks']==36720
    assert verify['nominal_interval_checks']==306 and verify['primary_family_interval_checks']==32
    lines=['# No-Early/no-Late DREChannNet: complete17-group selected-seed ablations','',
      'The user fixed seeds42,44,62 and promoted the existing six-feature no-Early/no-Late graph AFTER viewing historical Outer outcomes. This is outcome-informed seed-subset sensitivity on exploratory repeated-test patients, not random-seed robustness or independent confirmation. The unselected historical43/52main results are retained below; they are not hidden or merged into the requested primary estimate.','',
      'All255model choices froze before new Outer; all255scores sealed before metrics. Exactly50strict reuses and205fresh fits. Each group/seed retains80patients,7635channels,256seizures and1471965validRaw250windows (physiology-only consumes no Raw). Variance and Head Interaction remain absent. Same patient roles, loss, optimizer, FP32/TF32off/chunk4096/full-patient updates, original validation-only epoch/threshold rules, no performance gate or post-Outer tuning.','',
      '## Patient-equal means across the requested three seeds','',
      '| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table('MAIN_RESULTS'):lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','[Each seed](SEED_RESULTS.csv), [seed sample SD](SEED_SD.csv), [all255folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv). These average endpoint values, not probabilities, best seeds or best folds. EZ-fraction bias is predicted minus observed EZ fraction. Sources below5patients have suppressed metrics.','',
      '## Comparisons','',
      '10000seed42draws of the same80unique patients AFTER averaging their three seed metrics.17absolute+16Full-minus+1PRfactorial interaction yield306nominal intervals;32additional Bonferroni-percentile intervals cover paired Macro-F1/AUROC,tail0.05/64. Intervals are conditional on this outcome-informed selected seed set.','',
      '| Contrast | Delta Macro-F1 | Delta AUROC | Seeds both positive | Fold-seed cells both positive |',
      '| --- | ---: | ---: | ---: | ---: |']
    for r in table('COMPONENT_EVIDENCE'):lines.append(f"| {r['contrast']} | {float(r['macro_f1_difference']):+.6f} | {float(r['auroc_difference']):+.6f} | {r['seeds_both_positive']}/3 | {r['fold_seed_cells_both_positive']}/15 |")
    lines+=['','[All nominal intervals](INTERVALS.csv), [family intervals](PRIMARY_FAMILY_INTERVALS.csv), [all fold-seed directions](FOLD_SEED_DIRECTIONS.csv). Negative directions are retained. Restored-Early and other structural controls are not deletions: negative Full-minus-restored means the restored model did better.','',
      '## What the fixed controls isolate','',
      '- WO_BOTH_PR removes both direct physiological patientZ and finalRaw96DpatientZ. Ordinary waveform normalization and own pre-reference remain. WindowPR has no operative path in the no-guidance main and is only an alias, not another fit/hypothesis.',
      '- Six single deletions mask two direct summary coordinates BEFORE patient-relative statistics. With no guidance modules, no hidden window-conditioning route remains. A beneficial deletion is reported as such; these tests do not establish a globally optimal subset.',
      '- Raw-only/physiology-only keep original untrained head slices, scales and64hidden width, not compensation by widening.',
      '- EARLY_FILM_RESTORED restores the exact old no-Late model:10377parameters versus main9329. CAPACITY_MATCHED uses ordinary noEarly head73:10337parameters,40fewer than restoredEarly(-0.39%). It controls approximate parameter count, not all function classes or compute.',
      '- P1 changes both partitioning and within-seizure statistic: whole0-20meanSD then equal-seizure meanSD96D. It is not a pure segmentation-only test. P2 retains stages but adds26time-aware attention coefficients:9355parameters; comparisons include this capacity increment. Both pooling arms have no FiLM/Gate. Unsupported scorer inputs are masked BEFORE matmul to prevent NaN-gradient contamination; no valid-window removal.',
      '- No meaningful FiLM mismatch or Early/Late factorial interaction exists after both modules are absent. Their deletion aliases are documented without duplicate patient rows or hypotheses.','',
      '## Previously viewed main seeds: complete disclosure','',
      '| Existing seed | Included in requested subset | Macro-F1 | AUROC | AP | EZ-F1 |',
      '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in table('SEED_SELECTION_AUDIT'):lines.append('| '+r['seed']+' | '+r['included_in_requested_subset']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in ['macro_f1','auroc','auprc','f1'])+' |')
    goal=json.loads((PUB/'GOAL_AUDIT.json').read_text())
    lines+=['',f"Selected3main Macro-F1{goal['selected3_mean']['macro_f1']:.6f}; descriptive all5historical-main average{goal['all5_historical_main_mean']['macro_f1']:.6f}. The historical five-seed figure is not a newly run five-seed ablation. No new arm/seed is selected from this table.",'',
      f"Independent PASS:4080patient-seed rows,36720metric values,306nominal/32family intervals; maximum discrepancy{verify['maximum_interval_difference']:.3g}. Source/init/reuse/preprocessing/validation proofs and all completion records accompany the tables. Signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain private.",'']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert len(ALLOW)==36 and len(set(ALLOW))==36
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for n in ALLOW:
        p=PUB/n;assert p.is_file() and p.stat().st_size<5000000
        text=p.read_text(encoding='utf-8');assert not any(s in text for s in ['password=','PRIVATE KEY','fit_patient_ids'])
        if n.endswith('.csv'):
            with p.open(encoding='utf-8',newline='') as f:assert not set(next(csv.reader(f)))&forbidden
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in ALLOW:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(bytes=out.stat().st_size,sha256=sha(out),files=37)),flush=True)
if __name__=='__main__':main()
