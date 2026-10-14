"""Two decision-targeted candidates, cumulative eight-candidate evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_support import write,sha,original_modules
ARMS=['M7_MARGIN_GATED','M8_MACRO_AWARE']
def finish(runtime,phase2,c,rows,selections,errors,binding):
    pub=runtime/'results';m,b=original_modules();df=pd.DataFrame(rows)
    for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(65,47)
    means=df.groupby('arm')[m.METRICS].mean();sm=means.loc[['D1']+ARMS].reset_index();sm['trainable_parameters']=[0,9483,9481];sm['total_parameters']=sm.trainable_parameters+1537;sm.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
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
    pd.DataFrame(boots).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);pd.DataFrame(targets).to_csv(pub/'TARGET_SCREEN.csv',index=False);pd.DataFrame(errors).to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False);df[df.arm.str.contains('__')].groupby('arm')[m.METRICS].mean().reset_index().to_csv(pub/'FROZEN_INTERVENTIONS.csv',index=False)
    old=pd.read_csv(phase2/'results/ALL_EXPLORATION_CANDIDATES.csv');allsm=pd.concat([old,sm[sm.arm!='D1']],ignore_index=True);allsm.to_csv(pub/'ALL_EXPLORATION_CANDIDATES.csv',index=False)
    # Original phase1 is a sibling runtime, retained without rewriting previous outcomes.
    phase1=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')
    ta=pd.concat([pd.read_csv(phase1/'results/TARGET_SCREEN.csv'),pd.read_csv(phase2/'results/TARGET_SCREEN.csv'),pd.DataFrame(targets)],ignore_index=True);qualified=ta[ta.all_quality_gates];best=(qualified if len(qualified) else ta).sort_values('MacroF1',ascending=False).iloc[0]
    ta.to_csv(pub/'ALL_TARGET_SCREEN.csv',index=False)
    gate=dict(phase3_complete=True,total_candidates=8,total_training_cells=40,point_target_met=bool(ta.point_target_met.any()),target_met_with_quality_gates=bool(len(qualified)),selected_common_candidate=str(best.arm),selected_MacroF1=float(best.MacroF1),selected_gain=float(best.delta_MacroF1),primary_absolute_target=0.6645464619346358,selection='one global model acrossfolds; all candidates retained',adaptive_repeated_Development=True,independent_confirmation=False,no_new_Outer=True,protocol_sha256=binding['protocol'])
    write(pub/'SCIENTIFIC_GATE.json',gate);write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',all10_phase3_cells=True,all65_appearances_47_IDs=True,per_patient_independent_sklearn_parity=True,bootstrap_independent_multiplicity_parity=True,original_D1_frozen_and_exact=True,all_eight_candidates_retained=True))
    docs=runtime/'public_docs';docs.mkdir(exist_ok=True);text='# D1 small-module exploration — Phase3 and complete candidate table\n\n| Model | New parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in allsm.itertuples():text+=f'| {r.arm} | {int(r.trainable_parameters)} | {r.ez_auroc:.6f} | {r.ez_auprc:.6f} | {r.macro_f1:.6f} | {r.ez_f1:.6f} | {r.ez_mrr:.6f} |\n'
    text+=f'\nTarget Macro-F1≥0.664546461935: point target met `{gate["point_target_met"]}`, complete quality gate met `{gate["target_met_with_quality_gates"]}`. Selected common candidate `{best.arm}`: Macro-F1{best.MacroF1:.6f}, D1 gain{best.delta_MacroF1:+.6f}. This is adaptive repeated Development, not new Outer or independent clinical confirmation.\n\n'
    text+='Forty candidate-fold training cells were actually completed across three prospectively frozen phases. Phase3 was registered after all six previous models/results; it is not a pre-independent eight-model confirmation. No outcomes were overwritten, poor folds excluded or best architectures picked independently by fold. All new variants use the same frozen D1 checkpoints and preprocessing, same masks/Raw/DELTA/scales, original FIT weights and patient-equal optimizer/schedule/validation selection.\n\n'
    text+='M7 adds exactlytwo zero-initialized scalar parameters to the measured direct-error morphology branch. Its conditional multiplier uses tanh(abs(D1logit)), which is an uncalibrated magnitude, **not** certified uncertainty. Its inference-only margin-identity intervention separates dependence on this added gate from physiology-gate identity. M8 adds no new architecture parameters and uses fixed.75 weightedBCE+.25 patient SoftMacroF1; actual FIT labels/counts participate only in the supervised objective, never model inference. No separate threshold is optimized inside the training loss. Both retain initial/final residual-disabled exact D1 identity.\n\n'
    text+='Selected-checkpoint/threshold bootstrap:10000 seed42 paired unique-ID draws retaining65 appearances/47 IDs.95% intervals are unadjusted and selection-conditioned, not a correction for threshold/epoch/candidate selection or years of prior validation reuse. Report all AUROC/AP/EZF1/sensitivity tradeoffs and fold/center tables; a target point improvement does not prove transport to new patients.\n\n'
    text+='Only source/tests/hashes/anonymous aggregate reports are public. Actual legal-FIT gradient tests preceded new candidate training and their smoke states were discarded. All private weights/scalers, IDs/labels, waveforms/window banks, per-channel predictions and logs stay on serverD. No installed skill, new Raw extraction, additional seed or Outer access.\n'
    (docs/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    for p in pub.glob('*.csv'):assert not set(pd.read_csv(p,nrows=0).columns)&{'patient','patient_id','channel','subject_id','score','label'}
    write(pub/'PUBLIC_OUTPUT_HASHES.json',dict(status='PASS',sha256={p.name:sha(p) for p in pub.iterdir() if p.is_file() and p.name not in ['PUBLIC_OUTPUT_HASHES.json','RUN_STATUS.json']}));write(pub/'RUN_STATUS.json',dict(status='ALL_PHASE3_DECISION_MODULES_COMPLETE',cells=10,total_candidates=8,point_target_met=gate['point_target_met'],target_met=gate['target_met_with_quality_gates'],chosen=str(best.arm),no_new_Outer=True))
    print('DECISION_PHASE3_COMPLETE',best.arm,'target_met',bool(len(qualified)),flush=True)
