"""Explicit anonymous export; a diagnostic stage never completes the active goal."""
import zipfile
from common import *

ALLOW=['ADMISSION.json','ADMISSION_REUSE.csv','ROLE_AUDIT.csv','ANCHOR_ADMISSION.csv','ANCHOR_REPLAY.csv','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
       'CALIBRATION_REPLAY.csv','SELECTION_VERIFICATION.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv',
       'SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','PR_COMPONENT_EVIDENCE.csv',
       'POOLED_RESULTS.csv','QUALIFICATION_GATE.json','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT.md','COMPLETION_STATUS.json']

def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    binding,*_=setup();frozen();assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'ADMISSION.json')['source']==sources()
    mainrows=table('MAIN_RESULTS');evidence=table('PR_COMPONENT_EVIDENCE')
    gate=jr(PUB/'QUALIFICATION_GATE.json')
    lines=['# Six-feature DREChannNet branch-wise nonlinear PR residual: actual internal results','',
           'Exploratory repeated-test. These are the33-unique-patient, three-seed within-original-FIT ASSESS results, NOT a new80-patient Outer. The original main model Outer Macro-F1 remains0.653828; no direct performance difference against that point is estimated here.','',
           '## Execution and controls','',
           'Three seeds42/43/44 and all original five fold paths;15 exact internal B0 cells verified/reused and75 new full-patient fits. B0 has11425parameters; all five new variants have11553, with128zero-initialized gain parameters and the same single64-hidden head. EarlyFiLM, lateGate, six features, pre-reference, loss/budget and complete window/seizure pooling are unchanged.','',
           'The original hidden state is preserved at initialization. Bounded branch-wise patient-Z residuals are formed after GELU and added before the same final scalar classifier. PR deletion removes both original summary PR and the new nonlinear branch PR for that branch; a parameter-free per-channel LayerNorm replaces only the new residual normalization, never comparing other channels. Feature PR off uses TRAIN-only global imputation/scaling; Raw PR off uses a fixed per-coordinate affine preconditioner fitted under the matched initial encoder on all TRAIN channels. It is not claimed to be a current-encoder global reference or exact distribution matching during training.','',
           'CHANNEL_CONTEXT keeps both original summary PR operations but replaces the new residual context with per-channel LayerNorm, matching added capacity. It is not a no-PR model. All15 initial TRAIN raw anchors were independently replayed from original-model summaries, with no CAL/ASSESS input or labels.','',
           'All90 selections replayed/frozen before new assessment scores; all90 scores sealed before metrics. Prior B0 scores reused only after exact index/checkpoint/preprocessor/threshold checks. No new Outer, extra candidate/seed/loss/threshold search or post-assessment edits.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in mainrows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','## Separate PR evidence','',
            '| Contrast | Seeds with AUROC and Macro-F1 both positive | Stable positive points | Both nominal lower bounds positive | Both8-comparison lower bounds positive |',
            '| --- | ---: | --- | --- | --- |']
    for r in evidence:lines.append('| '+' | '.join(r[k] for k in ['contrast','seeds_both_positive','stable_positive_point_evidence','both_nominal_lowers_positive','both_family_lowers_positive'])+' |')
    lines+=['','See all108 nominal intervals, eight additional primary-family intervals and the PR difference-of-differences interaction in INTERVALS.csv. Zero-crossing intervals do not establish absence or superiority. Nonlinear training, normalization/reference differences and numeric scale remain part of each conditional comparison; an interaction is not proof of a physiological causal mechanism.','',
            '## Fixed next-stage qualification','',
            f"Performance margin: {gate['performance_margin_pass']}; both separate PR point rules: {gate['both_separate_PR_point_rules_pass']}; matched context rule: {gate['capacity_context_point_rule_pass']}; qualify for a separate core-module study: {gate['qualified_for_separate_core_module_study']}.",
            'Qualification uses predeclared internal point rules, not proof of statistical noninferiority or improvement. It never authorizes new Outer by itself. A failing group is not substituted post hoc as the main model.','',
            'Primary estimates average repeated fold metrics within unique patients, then metrics across three seeds, then patients equally. Pooled appearances are separate. Source summaries with fewer than5patients are suppressed. Bootstrap draws10000unique patients, retaining all paths and seeds paired; intervals are conditional on historical selection and do not cover full training randomness. Reusing the historical internal assessment does not create independent confirmation.','',
            '## Active objective remains open','',
            'This internal study cannot prove final80-patient five-fold Macro-F1>0.64 or contributions from all retained core modules. If qualified, separate prespecified EarlyFiLM/lateGate/pre-reference checks and final original-protocol fits/freeze/Outer evaluation are still required. If not qualified, report the negative evidence without weakening controls or relaxing gates. Previous temporal-contrast and PR-factorial results remain unchanged.','',
            'Raw signals, identities, individual predictions, checkpoints, preprocessors, caches and runtime logs remain on the trusted server. Only source, hashes and anonymous aggregates are exported.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='BRANCH_PR_INTERNAL_COMPLETE_VERIFIED',fresh_fits=75,exact_reused_fits=15,new_Outer=False,active_goal_achieved=False,
                                           qualified_for_separate_core_module_study=gate['qualified_for_separate_core_module_study'],next_step='If qualified, separately freeze core-module study; no best-Outer selection. Otherwise preserve negative evidence and do not promote this candidate.'))
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
