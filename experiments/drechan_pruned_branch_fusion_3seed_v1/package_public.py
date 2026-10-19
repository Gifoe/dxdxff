"""Explicit anonymous internal-stage export; never exports private artifacts."""
import zipfile
from common import *

ALLOW=['ADMISSION.json','ADMISSION_REUSE.csv','ROLE_AUDIT.csv','ANCHOR_ADMISSION.csv','ANCHOR_REPLAY.csv','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
       'CALIBRATION_REPLAY.csv','SELECTION_VERIFICATION.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv',
       'SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv','COMPONENT_EVIDENCE.csv',
       'POOLED_RESULTS.csv','QUALIFICATION_GATE.json','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT.md','COMPLETION_STATUS.json']


def main():
    setup();frozen();assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS' and jr(PUB/'ADMISSION.json')['source']==sources()
    with (PUB/'MAIN_RESULTS.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    gate=jr(PUB/'QUALIFICATION_GATE.json')
    lines=['# Pruned single-head branch fusion: actual internal results','',
           'Exploratory repeated-test:33unique within-FIT ASSESS patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828 is unchanged and is not directly compared to these internal values.','',
           '## Fixed comparison','',
           'B0 is the original11425parameter structure. JOINT_PRUNED removes EarlyFiLM/lateGate with original jointGELU. FULL and itsPR/pre-reference controls use separate branchGELUs summed with fixed1/sqrt(2),then the same one64-hidden/one-scalar head. All6new variants9329parameters; no new hiddenPR or residualgain. Pruning changes capacity; fusion and individual PR/pre-reference contrasts match parameter count.','',
           'FeaturePR off retains both modalities and uses TRAIN-only global imputation/scaling. RawPR off retains rawsummaries with a frozen complete-TRAIN initial-encoder affine. These are not zeroed inputs or inference-only deletions. The affine does not track current encoder distribution. All controls were retrained from matched original initialization.','',
           '90fresh fits and15verified original internalB0reuses. Same original five paths,seeds42/43/44,complete patients/raw windows,loss/optimizer/budget andCAL-only selection. All105selected cells replayed/frozen and all105scores sealed before metrics; no source/threshold/arm/seed change from these outcomes.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','Primary estimates average repeated folds within patient/seed,then all3seed metrics,then patients equally. Pooled appearances,fold/seedSD,source and coverage are separate. Sources<5patients are suppressed,not dropped.10000unique-patient paired bootstrap preserves all paths/seeds;135nominal and10predeclared additional intervals are conditional on historical selection and do not cover full training randomness.','',
            '## Fixed qualification','',json.dumps(gate,sort_keys=True),'',
            'Positive seed directions alone do not establish statistical support; inspect paired intervals. OnlyFULL is the candidate. An apparently better control cannot be substituted post hoc. Passing point rules permits only planning final original-protocol work,not automatically readingOuter. Final80-patient Macro-F1>0.64,AUROC improvement and retained-module contributions remain unproven by this stage. Failed findings remain unchanged; no relaxed gates or weakened controls.','',
            'This simplification was motivated by the previously completed negative residual/core studies,not an independent validation. No biological causality or globally optimal feature set is claimed.','',
            'Only source,hashes and anonymous aggregates are public. Raw,identities,individual scores,anchors,preprocessors,weights,caches and runtime logs remain private.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='PRUNED_FUSION_INTERNAL_COMPLETE_VERIFIED',fresh_fits=90,exact_reused_fits=15,new_Outer=False,active_goal_achieved=False,
                                           qualified_for_final_study_planning=gate['qualified_for_final_study_planning']))
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={name:sha(PUB/name) for name in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ALLOW:z.write(PUB/name,name)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(ALLOW)+1)),flush=True)


if __name__=='__main__':main()
