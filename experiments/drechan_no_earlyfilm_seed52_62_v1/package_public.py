"""Explicit anonymous aggregate release only after independent160-row PASS."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(r'D:\drechan_no_earlyfilm_seed52_62_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','ROLE_AUDIT.csv','INITIALIZATION_AUDIT.csv','SOURCE_PUSH_BEFORE_TRAIN.json',
 'TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','INTERNAL_VERIFICATION.json','VALIDATION_REPLAY_CHECKS.csv',
 'PREPROCESSING_REPLAY.csv','FREEZE_STATUS.json','PREDICTION_STATUS.json','RAW_COVERAGE.csv','MAIN_RESULTS.csv',
 'SEED_RESULTS.csv','SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','SOURCE_SEED_RESULTS.csv',
 'METRIC_COVERAGE.csv','INTERVALS.csv','POOLED_RESULTS.csv','POOLED_SEED_MEAN.csv','REPORT_STATUS.json',
 'INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','REPORT.md']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    assert json.loads((PUB/'OUTER_STATUS.json').read_text())['status']=='COMPLETE_VERIFIED_NO_EARLY_SEEDS52_62_FIVE_FOLD_OUTER'
    verify=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text())
    assert verify['status']=='PASS' and verify['patient_rows']==160 and verify['patient_metric_checks']==1440 and verify['nominal_interval_checks']==27
    lines=['# No Early FiLM and no Late Gate: seeds52 and62','',
      'Ten freshly initialized fold models on the original80patients/7635channels/256seizures/1471965validRaw250windows. Same five patient-held-out folds, corrected direct FeaturePR, final RawPR, own pre-ictal latent reference, original temporal/seizure pooling and head64. Variance and Head Interaction remain absent;9329parameters. No attention, widened head, mismatch control, new data or Full model at these two seeds.','',
      'All10models,preprocessors,epochs and validation-only thresholds froze before any new Outer inference; all10score cells sealed before metric reporting. No performance gate, budget or threshold tuning. This is exploratory repeated-test, not independent confirmation. Neither seed is selected as best.','',
      '| Seed/estimate | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table('SEED_RESULTS'):lines.append('| '+r['seed']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    for label,name in [('Two-seed metric mean','MAIN_RESULTS'),('Seed sample SD (n=2)','SEED_SD')]:
        r=table(name)[0];lines.append('| '+label+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','Primary endpoints are equal-patient averages, not pooled-channel metrics, a score ensemble or the best fold/seed. EZ-fraction bias is predicted minus observed EZ fraction; closer to zero is desirable.','',
      '[Ten folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [per-seed sources](SOURCE_SEED_RESULTS.csv), [pooled metrics separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv). Centers with fewer than5patients have suppressed metrics.','',
      'The27nominal95%patient-bootstrap intervals cover9endpoints for each seed and for the per-patient two-seed metric average. Resampling uses10000seed42draws of the same80unique patients, not160independent seed-patient rows. No superiority testing or multiplicity-adjusted comparison family is defined. Two seeds provide limited information about seed variability.','',
      'No matched Full models were run at52/62; these results alone cannot estimate a seed-matched Early FiLM contribution. Existing42/43/44 results are historical references and are not merged into the primary two-seed estimate.','',
      f"Independent reconstruction passed160patient rows,1440metric values and27intervals; maximum interval difference{verify['maximum_interval_difference']:.3g}. All scientific sources are hash-bound to pre-fit admission. Signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain private.",'']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    assert len(ALLOW)==28 and len(set(ALLOW))==28
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
    print(json.dumps(dict(bytes=out.stat().st_size,sha256=sha(out),files=29)),flush=True)
if __name__=='__main__':main()
