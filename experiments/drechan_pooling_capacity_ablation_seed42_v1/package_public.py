"""Export only explicit anonymous aggregates after independent final PASS."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','ROLE_AUDIT.csv','ADMISSION_REUSE.csv','INITIALIZATION_AUDIT.csv',
 'SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','INTERNAL_VERIFICATION.json',
 'VALIDATION_REPLAY_CHECKS.csv','PREPROCESSING_REPLAY.csv','FREEZE_STATUS.json','PREDICTION_STATUS.json','RAW_COVERAGE.csv',
 'MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv',
 'PRIMARY_FAMILY_INTERVALS.csv','COMPONENT_EVIDENCE.csv','FOLD_DIRECTIONS.csv','GOAL_AUDIT.json','POOLED_RESULTS.csv',
 'REPORT_STATUS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','REPORT.md']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    assert json.loads((PUB/'OUTER_STATUS.json').read_text())['status']=='COMPLETE_VERIFIED_PCA_5_GROUP_SEED42_FIVE_FOLD_OUTER'
    verify=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text());assert verify['status']=='PASS'
    lines=['# Pooling and capacity ablations: fixed no-Late DREChannNet, seed42','',
      'Actual original80-patient five-fold Outer,25cells:10strict reuses and15fresh fits. Variance, Head Interaction and Late Gate remain absent. All25choices froze before newOuter; all25scores sealed before reporting. Patient-equal metrics, not pooled channels or the best fold. This is exploratory repeated-test following historical Outer viewing, not independent confirmation. No result changes the fixed main.','',
      '## Operations fixed before fitting','',
      '- P1:0-20s valid-window population Mean/SD followed by equal-seizure Mean/SD,96D. This changes both temporal partition and the within-seizure statistic; it is not a pure isolation of segmentation.',
      '- P2:masked learned time-aware weighting separately in0-8s/8-20s, then original equal-seizure Mean/SD96D. Shared26coefficients on24latent,t/20,(t/20)^2,zero initialization; no labels/physiology in scorer.10403parameters versus Full10377:capacity difference is explicit.',
      '- A2:noEarlyFiLM, ordinary head widened64to73;10337parameters(-40,-0.39% versus Full10377). Original noEarly has9329. Shared64head neurons retain original untrained weights;9extra effective neurons use an independent fixed initializer. Direct physiology and finalPR remain; no physiological CNN modulation. This controls approximate parameter count, not exact functional capacity or compute.',
      '- Same complete-patient FP32/TF32off/chunk4096 updates,seed42 and original optimizer/loss/validation-only selection; no new data/features/extraction/search/performance gate. All groups complete Outer.','',
      '| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table('MAIN_RESULTS'):lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','EZ-fraction bias is predicted minus true EZ fraction; closeness to zero is desirable. [Folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).','',
      '## Paired comparisons','',
      '10000seed42patient draws;5absolute+4Full-minus+1A2-minus-unmatched-noEarly yield90nominal intervals. Ten additional Bonferroni-percentile intervals cover five paired Macro-F1/AUROC contrasts,tail0.05/20. No seedSD or training-seed robustness is implied.','',
      '| Contrast | Delta Macro-F1 | Delta AUROC | Folds both positive | Family lower bounds: Macro-F1,AUROC |',
      '| --- | ---: | ---: | ---: | --- |']
    for r in table('COMPONENT_EVIDENCE'):lines.append(f"| {r['contrast']} | {float(r['macro_f1_difference']):+.6f} | {float(r['auroc_difference']):+.6f} | {r['folds_both_positive']}/5 | {float(r['macro_family_low']):+.6f}, {float(r['auroc_family_low']):+.6f} |")
    lines+=['','[All90 nominal intervals](INTERVALS.csv) · [10family intervals](PRIMARY_FAMILY_INTERVALS.csv) · [fold directions](FOLD_DIRECTIONS.csv)','',
      'A2-minus-unmatched-noEarly isolates this particular ordinary-head capacity increase. Full-minus-A2 compares physiological modulation with approximately equal parameter count; it does not rule out every ordinary CNN/head architecture. P2 differences also include26attention parameters, so a win cannot be attributed exclusively to temporal selectivity. P1/P2 do not change the12active physiological summary coordinates or their preprocessing.','',
      f"Independent verification: {verify['patient_rows']}patient rows,{verify['patient_metric_checks']}metric values,90nominal/10family intervals; maximum interval discrepancy{verify['maximum_interval_difference']:.3g}.",
      'Only code/hashes/anonymous aggregates are public; signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    assert len(ALLOW)==29
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        text=p.read_text(encoding='utf-8');assert not any(x in text for x in ['password=','PRIVATE KEY','fit_patient_ids'])
        if name.endswith('.csv'):
            with p.open(encoding='utf-8',newline='') as f:assert not set(next(csv.reader(f)))&forbidden
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in ALLOW:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(bytes=out.stat().st_size,sha256=sha(out),files=30)),flush=True)
if __name__=='__main__':main()
