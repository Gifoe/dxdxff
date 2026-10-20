"""Explicit anonymous export; internal completion is not active-goal achievement."""
import zipfile
from common import *

ALLOW=['ADMISSION.json','ADMISSION_REUSE.csv','ROLE_AUDIT.csv','ANCHOR_ADMISSION.csv','ANCHOR_REPLAY.csv',
       'PREPROCESSING_ADMISSION.csv','PREPROCESSING_REPLAY.csv','NULL_CONTROL_RETRAIN_PARITY.csv',
       'SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','CALIBRATION_REPLAY.csv',
       'SELECTION_VERIFICATION.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','COVERAGE.csv','MAIN_RESULTS.csv','SEED_RESULTS.csv',
       'SEED_SD.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv',
       'COMPONENT_EVIDENCE.csv','POOLED_RESULTS.csv','QUALIFICATION_GATE.json','REPORT_STATUS.json','INDEPENDENT_VERIFICATION.json','REPORT.md','COMPLETION_STATUS.json']


def main():
    setup();frozen();assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS' and jr(PUB/'ADMISSION.json')['source']==sources()
    with (PUB/'MAIN_RESULTS.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    gate=jr(PUB/'QUALIFICATION_GATE.json')
    lines=['# Unit-consistent FeaturePR:actual internal experiment','',
           'Exploratory repeated-test:33unique within-original-FIT assessment patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828remains unchanged,not a guaranteed post-correction score or a comparable internal endpoint.','',
           '## Exact scope','',
           '90fresh fits plus45exact reference reuses under original five roles,seeds42/43/44,full patients/windows/loss/AdamW/precision/budgets andCAL-only epoch/threshold selection. All135selections replayed/frozen andall135score cells sealed before unified metrics.','',
           'Only direct FeaturePR numerical definition changes. TRAIN-only raw imputer/StandardScaler provides per-coordinate units. Patient SD is compared with1e-8times that TRAINunit;valid coordinates use centered/rawSD anddegenerate observed coordinates become0,missing values stayNaN. The original second imputer/StandardScaler is fitted onTRAIN only. All6physiological features remain,with only the oldVariance pair masked. No clipping,patient exclusion,feature search oroutcome-tuned epsilon.','',
           'The NoFeaturePR matrix/scaler is the original TRAIN-only global input,bit unchanged. AllNoFeaturePR/bothPR models were nevertheless freshly retrained andindependently required to match the prior identical tensor procedure. A fair control is not made weaker by a new scale or missing branch.','',
           'The dual-view CNN/96Dpooling/192DRawinput/14widephysiology/one64-hidden jointGELU/scalar head,RawPR,pre-reference andfixed initialTRAIN Rawanchors are unchanged. No evolving-encoder anchor refresh,loss ordecoder rescue is added. All6new arms have15473parameters;referenceB0has11425,PRUNED_REF9329,andLEGACY_DUALVIEW15473. Reference-only gains cannot isolate added capacity;matched module controls are separately reported.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','Primary estimates average repeated folds within patient/seed,then3seed metrics,then patients equally. Separate pooled,source/fold/seedSD andcoverage tables are provided. Source<5unique patients suppressed,not dropped.10000unique-patient seed42paired bootstrap gives162nominal and14predeclared additional-family intervals,conditional on models andhistorically adaptive research,not independent confirmation or full training uncertainty.','',
            '## Fixed qualification','',json.dumps(gate,sort_keys=True),'',
            'Seed directions andpoint rules do not establish statistical stability. Qualification requires the main performance andretained-module point rules;UNIT_CONSISTENCY against the old mixed-unit procedure is reported as a diagnostic comparison,not proof that a mathematical correction must improve prediction. Its nominal/additional intervals are not omitted on a negative result. No better control is promoted after outcomes.','',
            'Internal completion orqualification does not achieve final80-patient Macro-F1>0.64,improvedAUROC orstable retained-module contributions anddoes not automatically authorize newOuter. Actual final original-protocol fitting/freeze/evaluation remains required. No model,threshold,comparison family,patient orseed changed after newassessment.','',
            'Only code,hashes andanonymous aggregates are public. Raw,identities,individual predictions,weights,preprocessors,anchors,caches andlogs remain on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='UNIT_CONSISTENT_PR_INTERNAL_COMPLETE_VERIFIED',fresh_fits=90,exact_reused_fits=45,
                                          new_Outer=False,active_goal_achieved=False,qualified_for_final_study_planning=gate['qualified_for_final_study_planning']))
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    assert len(ALLOW)==len(set(ALLOW))==33
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:assert not set(next(csv.reader(f)))&forbidden
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    target=ROOT/'public_aggregate_export_v1.zip'
    if target.exists():
        with zipfile.ZipFile(target) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in ALLOW:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target),files=len(ALLOW)+1)),flush=True)


if __name__=='__main__':main()
