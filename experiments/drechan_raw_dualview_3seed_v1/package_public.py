"""Explicit anonymous export of a verified internal study,not goal completion."""
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
    lines=['# Global-plus-relative Raw dual view: actual internal results','',
           'Exploratory repeated-test:33unique within-original-FIT ASSESS patients,three seeds,not new80-patientOuter. Historical mainOuter Macro-F1=0.653828 remains unchanged and is not directly subtracted from these internal values.','',
           '## Fixed architecture and controls','',
           'B0andPRUNED_REF are exact frozen parent references,not newly selected winners.90fresh fits and30reference reuses,same original five paths,seeds42/43/44,full patients/windows/loss/optimization/CAL-only selection. All120selections replayed/frozen and all120score cells sealed before metrics.','',
           'Each new arm has15473parameters,one64-hidden jointGELU and one scalar classifier. The original unconditioned24D CNN/pre-reference/stage/seizure pooling gives96D v. FULL concatenates TRAIN-global affine g(v) and current patientZ(v),then the same six-feature14wide masked summary. EarlyFiLM,lateGate,old residualgains and extra hiddenPR remain absent. B0has11425parameters andPRUNED_REF9329; unconditional reference improvements alone cannot establish context-specific benefit.','',
           'RawPR deletion retains g and substitutes channelZ(g) for patientZ(v). Global-view deletion retains r=patientZ(v) and substitutes channelZ(r) for g. These are parameter-/dimension-matched nonzero within-channel alternatives,not necessarily identical function classes. FeaturePR deletion uses TRAIN-only global imputation/scaling,not deleted physiology. BothPR deletion combines them while retaining per-window waveform normalization and own-channel temporal reference.','',
           'The global view is a latent summary before final patientZ,not restored physical EEG amplitude. Its fixed initialTRAIN affine does not track the evolving encoder.30new pre/no-pre anchors were independently fitted and replayed on all genuine TRAIN windows without labels orCAL/ASSESS. The no-pre control fits its own legal scale; its contrast includes that necessary preprocessing refit,not only an isolated algebraic subtraction.','',
           '| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+r['arm']+' | '+' | '.join(f'{float(r[m]):.6f}' for m in METRICS)+' |')
    lines+=['','Primary estimates average repeated folds within patient/seed,then all3seed metrics,then patients equally. Source<5suppressed,not dropped;pooled appearances,seed/foldSD and coverage are separate.10000unique-patient paired bootstrap retains all paths/seeds.144nominal and12fixed additional intervals are conditional on historical research selection and do not cover full training randomness.','',
            '## Fixed next-stage rule','',json.dumps(gate,sort_keys=True),'',
            'Seed-direction consistency is not statistical proof:inspect nominal/additional intervals separately. OnlyFULL is the candidate. A better-looking control cannot be promoted post hoc. A point pass permits only planning final original-protocol work,not automatic newOuter or active-goal completion. Final80-patient Macro-F1>0.64,improvedAUROC and stable retained-module contributions still require actual evidence. Prior failed studies and their gates remain unchanged.','',
            'No newOutcome-driven source,threshold,cohort,seed or comparison-family changes. No independent-confirmation,biological-causality or global-optimality claim.','',
            'Only source,hashes and anonymous aggregates are public. Raw,identities,individual scores,anchors,preprocessors,weights,caches and runtime logs remain private.','']
    (PUB/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(PUB/'COMPLETION_STATUS.json',dict(status='RAW_DUALVIEW_INTERNAL_COMPLETE_VERIFIED',fresh_fits=90,exact_reused_fits=30,new_Outer=False,active_goal_achieved=False,
                                           qualified_for_final_study_planning=gate['qualified_for_final_study_planning']))
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip'
    if out.exists():
        with zipfile.ZipFile(out) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ALLOW:z.write(PUB/name,name)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(ALLOW)+1)),flush=True)


if __name__=='__main__':main()
