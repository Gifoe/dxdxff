"""Explicit anonymous export; a diagnostic stage never completes the active goal."""
import zipfile
from common import *

ALLOW=['ADMISSION.json','ADMISSION_REUSE.csv','ROLE_AUDIT.csv','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
       'CALIBRATION_REPLAY.csv','SELECTION_VERIFICATION.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv',
       'SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','PR_COMPONENT_EVIDENCE.csv',
       'POOLED_RESULTS.csv','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT.md','COMPLETION_STATUS.json']

def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    binding,*_=setup();frozen();assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'ADMISSION.json')['source']==sources()
    mainrows=table('MAIN_RESULTS');evidence=table('PR_COMPONENT_EVIDENCE')
    lines=['# Six-feature DREChannNet PR component factorial: actual internal results','',
           'Exploratory repeated-test. These are the33-unique-patient, three-seed within-original-FIT ASSESS results, NOT a new80-patient Outer. The original main model Outer Macro-F1 remains0.653828; no direct performance difference against that point is estimated here.','',
           '## Execution and controls','',
           'Three seeds42/43/44 and all original five fold paths;15 exact internal B0 cells verified/reused and45 new full-patient fits. All four arms have11425parameters, the same original budget/loss/initialization and a64-hidden single head. EarlyFiLM, lateGate, six features, pre-reference and complete window/seizure pooling are unchanged.','',
           'Feature PR off omits only patient Z, refitting the original imputation/scaling recipe on TRAIN only. Raw PR off omits only final96D patient Z. The two normalizations are independently varied; raw and physiological branch inputs are not artificially zeroed. Numeric-scale changes are part of this original deletion, so these contrasts alone do not isolate physiological information from scale/optimization.','',
           'All60 selections replayed/frozen before new assessment scores; all60 scores sealed before metrics. Prior B0 scores reused only after exact index/checkpoint/preprocessor/threshold checks. No new Outer, candidate threshold search or test-driven model edits.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in mainrows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','## Separate PR evidence','',
            '| Contrast | Seeds with AUROC and Macro-F1 both positive | Stable positive points | Both nominal lower bounds positive | Both4-comparison lower bounds positive |',
            '| --- | ---: | --- | --- | --- |']
    for r in evidence:lines.append('| '+' | '.join(r[k] for k in ['contrast','seeds_both_positive','stable_positive_point_evidence','both_nominal_lowers_positive','both_family_lowers_positive'])+' |')
    lines+=['','See all108 nominal intervals, four additional primary-family intervals and the difference-of-differences interaction in INTERVALS.csv. Zero-crossing intervals do not establish absence or superiority. In8 factorial contrasts, numerical-scale/calibration effects and nonlinear training can interact; an interaction is not proof of a physiological causal mechanism.','',
            'Primary estimates average repeated fold metrics within unique patients, then metrics across three seeds, then patients equally. Pooled appearances are separate. Source summaries with fewer than5patients are suppressed. Bootstrap draws10000unique patients, retaining all paths and seeds paired; intervals are conditional on historical selection and do not cover full training randomness. Reusing the historical internal assessment does not create independent confirmation.','',
            '## Active objective remains open','',
            'This diagnostic cannot prove that a new model has five-fold Macro-F1>0.64, improves AUROC, or has useful contributions from all retained modules. It identifies separate PR effects for subsequent predeclared optimization. No control is weakened or unfavorable result hidden to manufacture a PR advantage. The previous temporal-contrast failed-gate result remains unchanged.','',
            'Raw signals, identities, individual predictions, checkpoints, preprocessors, caches and runtime logs remain on the trusted server. Only source, hashes and anonymous aggregates are exported.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='DIAGNOSTIC_FACTORIAL_COMPLETE_VERIFIED',fresh_fits=45,exact_reused_fits=15,new_Outer=False,active_goal_achieved=False,
                                           next_step='Use internally observed separate PR/interaction evidence to predeclare the next architecture/normalization comparison; never tune from Outer.'))
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ALLOW:z.write(PUB/name,name)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(ALLOW)+1)),flush=True)

if __name__=='__main__':main()
