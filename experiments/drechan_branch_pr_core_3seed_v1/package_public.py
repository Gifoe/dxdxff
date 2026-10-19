"""Explicit anonymous export; a diagnostic stage never completes the active goal."""
import zipfile
from common import *

ALLOW=['ADMISSION.json','ADMISSION_BRIDGES.csv','ROLE_AUDIT.csv','SOURCE_PUSH_BEFORE_TRAIN.json','NEW_TRAIN_STATUS.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
       'CALIBRATION_REPLAY.csv','SELECTION_VERIFICATION.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv',
       'SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','CORE_COMPONENT_EVIDENCE.csv',
       'POOLED_RESULTS.csv','QUALIFICATION_GATE.json','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT.md','COMPLETION_STATUS.json']

def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    binding,*_=setup();frozen();assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'ADMISSION.json')['source']==sources()
    mainrows=table('MAIN_RESULTS');evidence=table('CORE_COMPONENT_EVIDENCE')
    gate=jr(PUB/'QUALIFICATION_GATE.json')
    lines=['# Same-candidate DREChannNet core-module audit: actual internal results','',
           'Exploratory repeated-test. These are the33-unique-patient, three-seed within-original-FIT ASSESS results, NOT a new80-patient Outer. The original main model Outer Macro-F1 remains0.653828; no direct performance difference against that point is estimated here.','',
           '## Execution and controls','',
           'Three seeds42/43/44 and all original five fold paths;15exact parent-candidate FULL cells verified/reused and75new complete-patient fits. This FULL is the branch-relative candidate, not the old B0 or original0.653828Outer model. Both PR operations, six-feature masks,128relative gains, one64-hidden head, loss/budget and full pooling remain unchanged across this core study.','',
           'Only EarlyFiLM, lateGate, both guidance operations together, or raw pre-reference is removed. Shared tensors and original full initializer RNG consumption are preserved. Parameters:11553FULL/pre-reference/mismatch,10505single-guidance deletion,9457both-guidance deletion. Capacity differences are explicit and prevent attributing deletion effects wholly to physiological information.','',
           'The early mismatch retains full capacity and changes only the early-conditioning donor trajectory within the same patient/seizure/identical complete valid-support group, without labels. Singleton support groups remain unmodified; changed/eligible/singleton/total coverage is reported. Correct lateGate/direct physiological inputs remain unchanged. This control is diagnostic, never a winner to promote.','',
           'All90selections replayed/frozen before NEW control assessment scores. The15FULL scores reuse the parent completed hash-bound seal; all90core score cells seal before unified metrics. No newOuter or post-assessment architecture/loss/threshold/seed/subject edits.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in mainrows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','## Retained-core contribution and early-context diagnostic','',
            '| Contrast | Seeds with AUROC and Macro-F1 both positive | Stable positive points | Both nominal lower bounds positive | Both8-comparison lower bounds positive |',
            '| --- | ---: | --- | --- | --- |']
    for r in evidence:lines.append('| '+' | '.join(r[k] for k in ['contrast','seeds_both_positive','stable_positive_point_evidence','both_nominal_lowers_positive','both_family_lowers_positive'])+' |')
    lines+=['','See all108 nominal intervals, eight additional primary-family intervals and the EarlyFiLM/LateGate difference-of-differences interaction in INTERVALS.csv. Zero-crossing intervals do not establish absence or superiority. Nonlinear training, conditioning and parameter-capacity differences remain part of each conditional deletion; an interaction is not proof of a physiological causal mechanism.','',
            '## Fixed next-stage qualification','',
            f"Parent fixed qualification: {gate['parent_fixed_rule_pass']}; all3retained single-core point rules: {gate['all_three_retained_core_point_rules_pass']}; correct early-context diagnostic: {gate['correct_early_context_diagnostic_pass']}; eligible to plan a final study: {gate['final_study_planning_eligible']}.",
            'These are fixed internal point rules, not statistical proof or authorization to read newOuter. A negative contribution is not hidden, repaired by weakening controls, or silently removed from this model. Any optimization/pruning is a separately frozen study; no best-Outer selection.','',
            'Primary estimates average repeated fold metrics within unique patients, then metrics across three seeds, then patients equally. Pooled appearances are separate. Source summaries with fewer than5patients are suppressed. Bootstrap draws10000unique patients, retaining all paths and seeds paired; intervals are conditional on historical selection and do not cover full training randomness. Reusing the historical internal assessment does not create independent confirmation.','',
            '## Active objective remains open','',
            'This internal study cannot prove final80-patient five-fold Macro-F1>0.64 or improved AUROC. Even if all point rules pass, original-protocol final fitting/freeze/Outer and paired uncertainty remain unfinished. If a retained module does not contribute, optimize or simplify it in a separately fixed comparison. Previous temporal-contrast, PR-factorial and parent-candidate results remain unchanged.','',
            'Raw signals, identities, individual predictions, checkpoints, preprocessors, caches and runtime logs remain on the trusted server. Only source, hashes and anonymous aggregates are exported.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='CORE_MODULE_INTERNAL_COMPLETE_VERIFIED',fresh_fits=75,exact_reused_fits=15,new_Outer=False,active_goal_achieved=False,
                                           final_study_planning_eligible=gate['final_study_planning_eligible'],next_step='Only if parent/core evidence qualifies, plan original-protocol final study. Otherwise separately optimize/prune; never alter these results or choose onOuter.'))
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
