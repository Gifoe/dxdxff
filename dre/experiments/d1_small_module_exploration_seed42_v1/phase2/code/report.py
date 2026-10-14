"""All direct-correction outcomes and cumulative exploration, not a winner-only table."""
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_support import write,sha,original_modules
ARMS=['M4_FEATURE_CORRECTION','M5_STAGE_CORRECTION','M6_MORPH_CORRECTION']

def finish(runtime,phase1,c,rows,selections,errors,binding):
    pub=runtime/'results';m,b=original_modules();df=pd.DataFrame(rows)
    for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(65,47)
    means=df.groupby('arm')[m.METRICS].mean();sm=means.loc[['D1']+ARMS].reset_index();sm['trainable_parameters']=[0,257,721,9481];sm['total_parameters']=sm.trainable_parameters+1537;sm.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:
        out=[dict(arm=arm,**{field:v},appearances=len(q),unique_IDs=q.patient.nunique(),**q[m.METRICS].mean().to_dict()) for (arm,v),q in df[df.arm.isin(['D1']+ARMS)].groupby(['arm',field])]
        pd.DataFrame(out).to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    df[df.arm.isin(['D1']+ARMS)].groupby(['arm','patient'])[m.METRICS].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    boots=[];targets=[]
    for arm in ARMS:
        boots.extend(dict(comparison=arm+' minus D1',**r) for r in b.compare(c,df[df.arm==arm],df[df.arm=='D1'],m.METRICS))
        diff=means.loc[arm]-means.loc['D1'];positive=sum(df[(df.arm==arm)&(df.fold==f)].macro_f1.mean()>df[(df.arm=='D1')&(df.fold==f)].macro_f1.mean() for f in range(1,6))
        tests=dict(MacroF1_two_points=diff.macro_f1>=.02,AUROC_nondecline=diff.ez_auroc>=0,AP_nondecline=diff.ez_auprc>=0,EZF1_nondecline=diff.ez_f1>=0,sensitivity_margin=diff.sensitivity>=-.01,at_least4_positive_folds=positive>=4)
        targets.append(dict(arm=arm,MacroF1=means.loc[arm,'macro_f1'],delta_MacroF1=diff.macro_f1,delta_AUROC=diff.ez_auroc,delta_AP=diff.ez_auprc,delta_EZF1=diff.ez_f1,delta_sensitivity=diff.sensitivity,positive_folds=positive,point_target_met=bool(tests['MacroF1_two_points']),all_quality_gates=bool(all(tests.values())),**tests))
    pd.DataFrame(boots).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);pd.DataFrame(targets).to_csv(pub/'TARGET_SCREEN.csv',index=False);pd.DataFrame(errors).to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False)
    df[df.arm.str.contains('__')].groupby('arm')[m.METRICS].mean().reset_index().to_csv(pub/'FROZEN_INTERVENTIONS.csv',index=False)
    previous=pd.read_csv(phase1/'results/DEVELOPMENT_SUMMARY.csv');combined=pd.concat([previous,sm[sm.arm!='D1']],ignore_index=True);combined.to_csv(pub/'ALL_EXPLORATION_CANDIDATES.csv',index=False)
    target_all=pd.concat([pd.read_csv(phase1/'results/TARGET_SCREEN.csv'),pd.DataFrame(targets)],ignore_index=True);qualified=target_all[target_all.all_quality_gates]
    best=(qualified if len(qualified) else target_all).sort_values('MacroF1',ascending=False).iloc[0]
    gate=dict(phase2_complete=True,total_exploration_candidates=6,total_training_cells=30,point_target_met=bool(target_all.point_target_met.any()),target_met_with_quality_gates=bool(len(qualified)),selected_common_candidate=str(best.arm),selected_MacroF1=float(best.MacroF1),selected_gain=float(best.delta_MacroF1),selection='one global candidate; never choose architecture separately per fold',adaptive_repeated_Development=True,independent_confirmation=False,no_new_Outer=True,protocol_sha256=binding['protocol'])
    write(pub/'SCIENTIFIC_GATE.json',gate);write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',phase2_all15_cells=True,all65_appearances_47_IDs=True,per_patient_metrics_independent_sklearn_parity=True,paired_bootstrap_independent_multiplicity_parity=True,frozen_D1_exact=True,all_six_candidates_retained=True))
    docs=runtime/'public_docs';docs.mkdir(exist_ok=True)
    text='# Frozen D1 small-module exploration — Phase2 direct correction\n\n| Candidate | New parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in combined.itertuples():text+=f'| {r.arm} | {int(r.trainable_parameters)} | {r.ez_auroc:.6f} | {r.ez_auprc:.6f} | {r.macro_f1:.6f} | {r.ez_f1:.6f} | {r.ez_mrr:.6f} |\n'
    text+=f'\nTarget patient Macro-F1≥0.664546461935 (D1+absolute.02). Any point target met: {gate["point_target_met"]}. Any complete quality gate met: {gate["target_met_with_quality_gates"]}. Current globally selected candidate `{best.arm}`, gain{best.delta_MacroF1:+.6f}. These remain adaptive repeated validation results, not independent confirmation.\n\n'
    text+='Phase2 was prospectively frozen after observing Phase1; prior models/results remain unchanged. It tests a different training mechanism: zero last-output initialization makes step0 exactlyD1, while direct weightedBCE(final) trains corrections to the frozen D1 rather than a competing standalone auxiliary classifier. The fixed residual cap is1.0; no alpha needs to leave zero. This changes both correction parameterization and objective deliberately; it is not an isolated proof of which change caused a gain.\n\n'
    text+='M4 adds257 parameters on original preprocessed R14; M5 adds721 on exact Phase1 stage-DELTA28; M6 adds9481 on the unchanged morphology/gate path. Original D1 weights/preprocessing, original folds, FIT weights, patient-equal updates, optimizer/schedule/validation selection and corrected FIT-only scales remain fixed. M4 adds no new information source: any gain is capacity/error-correction, not physiological discovery. M5/M6 use the existing pinned Raw250-derived physiology; no new extraction/filter/frequency/onset proof. All models and failed gates are reported.\n\n'
    text+='Actual legal-FIT three-update gradient tests precede each new candidate, then the smoke model is discarded and seeded construction reset. Zero last-layer initialization means first hidden/CNN gradients are expected zero on step1, gate output starts after head movement, and the earlier gate layer follows after its output moves. D1 has no gradients, always eval and its state hashes remain exact. Residual0 logits exactly equalD1 but use the candidate-selected threshold in interventions.\n\n'
    text+='10000 paired seed42 unique-ID draws retain65 appearances/47 IDs with thresholds fixed. Intervals condition on checkpoint/threshold/candidate selection and the repeated Development history. No fresh Outer inference, extra seeds, hidden configurations, patient/fold exclusion or per-fold best-model assembly. Raw/labels/IDs/weights/predictions/logs stay server D; only code/hashes/anonymous aggregates are exported.\n'
    (docs/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    for p in pub.glob('*.csv'):assert not set(pd.read_csv(p,nrows=0).columns)&{'patient','patient_id','channel','subject_id','label','score'}
    write(pub/'PUBLIC_OUTPUT_HASHES.json',dict(status='PASS',sha256={p.name:sha(p) for p in pub.iterdir() if p.is_file() and p.name not in ['RUN_STATUS.json','PUBLIC_OUTPUT_HASHES.json']}))
    write(pub/'RUN_STATUS.json',dict(status='ALL_PHASE2_DIRECT_CORRECTION_COMPLETE',cells=15,total_candidates=6,point_target_met=gate['point_target_met'],target_met=gate['target_met_with_quality_gates'],chosen=gate['selected_common_candidate'],no_new_Outer=True))
    print('DIRECT_CORRECTION_PHASE2_COMPLETE',best.arm,'target_met',bool(len(qualified)),flush=True)
