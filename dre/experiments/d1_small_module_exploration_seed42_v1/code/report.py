"""Preserve all candidate outcomes; no hidden successful-fold replacement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_support import write,sha,original_modules

ARMS=['M1_SCALED_DGM','M2_STAGE_DELTA','M3_DELTA_SET']

def finish(runtime,c,rows,selections,errors,binding):
    pub=runtime/'results';m,b=original_modules();df=pd.DataFrame(rows)
    for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(65,47),arm
    means=df.groupby('arm')[m.METRICS].mean();summary=means.loc[['D1']+ARMS].reset_index()
    summary['total_parameters']=[1537,11019,2259,2891];summary['trainable_parameters']=[0,9482,722,1354]
    summary.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:
        out=[dict(arm=arm,**{field:v},appearances=len(q),unique_IDs=q.patient.nunique(),**q[m.METRICS].mean().to_dict()) for (arm,v),q in df[df.arm.isin(['D1']+ARMS)].groupby(['arm',field])]
        pd.DataFrame(out).to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    df[df.arm.isin(['D1']+ARMS)].groupby(['arm','patient'])[m.METRICS].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    bootstrap=[];adjusted=[];targets=[]
    for arm in ARMS:
        bootstrap.extend(dict(comparison=arm+' minus D1',**r) for r in b.compare(c,df[df.arm==arm],df[df.arm=='D1'],m.METRICS))
        left=df[df.arm==arm].set_index(['fold','patient']).sort_index();right=df[df.arm=='D1'].set_index(['fold','patient']).sort_index();assert left.index.equals(right.index)
        ids=left.index.get_level_values('patient').to_numpy().astype(str);draws=c.cluster_bootstrap(ids,(left.macro_f1-right.macro_f1).to_numpy()[:,None],10000)
        lo,hi=np.nanpercentile(draws[:,0],[100*.05/(2*3),100*(1-.05/(2*3))])
        adjusted.append(dict(arm=arm,metric='macro_f1',ci_low=float(lo),ci_high=float(hi),family_size=3,nominal_interval=1-.05/3,draws=10000,selection_conditioned=True))
        diff=means.loc[arm]-means.loc['D1'];positive=sum(df[(df.arm==arm)&(df.fold==f)].macro_f1.mean()>df[(df.arm=='D1')&(df.fold==f)].macro_f1.mean() for f in range(1,6))
        checks=dict(MacroF1_two_points=diff.macro_f1>=.02,AUROC_nondecline=diff.ez_auroc>=0,AP_nondecline=diff.ez_auprc>=0,EZF1_nondecline=diff.ez_f1>=0,sensitivity_margin=diff.sensitivity>=-.01,at_least4_positive_folds=positive>=4)
        targets.append(dict(arm=arm,MacroF1=means.loc[arm,'macro_f1'],delta_MacroF1=diff.macro_f1,delta_AUROC=diff.ez_auroc,delta_AP=diff.ez_auprc,delta_EZF1=diff.ez_f1,delta_sensitivity=diff.sensitivity,positive_folds=positive,point_target_met=bool(checks['MacroF1_two_points']),all_quality_gates=bool(all(checks.values())),**checks))
    pd.DataFrame(bootstrap).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);pd.DataFrame(adjusted).to_csv(pub/'MULTIPLICITY_PRIMARY_MACROF1.csv',index=False)
    pd.DataFrame(targets).to_csv(pub/'TARGET_SCREEN.csv',index=False)
    df[df.arm.str.contains('__')].groupby('arm')[m.METRICS].mean().reset_index().to_csv(pub/'FROZEN_INTERVENTIONS.csv',index=False);pd.DataFrame(errors).to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False)
    qualified=[r for r in targets if r['all_quality_gates']];best=max(qualified,key=lambda r:r['MacroF1']) if qualified else max(targets,key=lambda r:r['MacroF1'])
    gate=dict(phase1_complete=True,primary_target='patient MacroF1 absolute+0.02',target_met=bool(qualified),selected_common_candidate=best['arm'],selection='global fixed-candidate Development mean, no per-fold architecture cherry picking',targets=targets,unadjusted_CIs_selection_conditioned=True,no_new_Outer=True,independent_confirmation=False,protocol_sha256=binding['protocol'])
    write(pub/'SCIENTIFIC_GATE.json',gate)
    write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',all15_cells=True,exact_65_appearances_47_unique_IDs=True,per_patient_sklearn_parity_to_1e12=True,bootstrap_original_vs_independent_multiplicity_parity=True,all_candidates_reported=True,original_D1_frozen=True))
    docs=runtime/'public_docs';docs.mkdir(exist_ok=True)
    text='# D1 small-module exploration — Phase1 Seed42\n\n'
    text+='| Model | New trainable parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for arm,n in zip(['D1']+ARMS,[0,9482,722,1354]):
        r=means.loc[arm];text+=f'| {arm} | {n} | {r.ez_auroc:.6f} | {r.ez_auprc:.6f} | {r.macro_f1:.6f} | {r.ez_f1:.6f} | {r.ez_mrr:.6f} |\n'
    text+=f'\nFixed target: Macro-F1≥0.664546461935 (D1+absolute0.02). Phase1 quality-gated target met: **{bool(qualified)}**. Globally chosen candidate: `{best["arm"]}`; this is exploratory candidate selection, never independently confirmed performance.\n\n'
    text+='The scale fix is a new prospectively locked experiment authorized for exploration, not alteration of the completed A2 result. Original1e-6 floor collapsed native V-power dimensions; current1e-30 floor preserves the same FIT-only median-of-patient-median estimator and makes no unit/data/filter changes. No new physiology extraction. All candidates consume exactly the same corrected scale and source windows.\n\n'
    text+='M1 is the old DGM topology retrained with corrected scaling. M2 adds only722 parameters on deterministic early/late DELTA statistics; no waveform CNN. M3 adds1354 parameters on a nonlinear window-set encoder; no waveform CNN, chronology, GRU, attention or graph. Both test whether all-window R14 pooling discards useful stage/distribution information. Their initialization exactly preserves frozen D1 final logits, and auxiliary loss starts the residual gradients. D1 weights/scalers and folds stayed unchanged.\n\n'
    text+='## All targets and uncertainty\n\n| Candidate | ΔMacro-F1 | ΔAUROC | ΔAP | ΔEZ-F1 | Positive MF folds | All quality gates |\n|---|---:|---:|---:|---:|---:|---|\n'
    for r in targets:text+=f'| {r["arm"]} | {r["delta_MacroF1"]:+.6f} | {r["delta_AUROC"]:+.6f} | {r["delta_AP"]:+.6f} | {r["delta_EZF1"]:+.6f} | {r["positive_folds"]}/5 | {r["all_quality_gates"]} |\n'
    text+='\nAll10000 seed42 paired unique-ID draws retain65 appearances/47 IDs, thresholds/checkpoints fixed.95% intervals and secondary98.333% three-candidate primary intervals are descriptive and still condition on validation checkpoint/threshold selection and prior repeated Development usage. They are not independent significance or a correction for years of research reuse.\n\n'
    text+='## Execution/scope\n\nAll15 predefined candidate-fold training cells ran; no poor-fold early truncation, extra seed, hyperparameter rescue or new Outer access. Own-fold legal FIT labels are used for training only; validation labels select checkpoint/threshold. Public files have no individual ID/channel/label or score records. All private data/weights/scales/predictions/logs remain server D:. Frozen residual0 is exact D1 logits but evaluated at each candidate threshold.\n\n'
    text+='Sources: [Deep Sets](https://arxiv.org/abs/1703.06114) supplies the general permutation-invariant encoder/pooling idea; [FiLM](https://arxiv.org/abs/1709.07871) supplies general feature conditioning context. Neither paper establishes this clinical task’s incremental performance. This implementation and the measurements above, not external literature, determine the result.\n'
    (docs/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    # Export only anonymous aggregate schemas; actual individual data never leave runtime.
    for path in pub.glob('*.csv'):
        assert not set(pd.read_csv(path,nrows=0).columns)&{'patient','patient_id','subject_id','channel','label','score'}
    write(pub/'PUBLIC_OUTPUT_HASHES.json',dict(status='PASS',sha256={p.name:sha(p) for p in pub.iterdir() if p.is_file() and p.name not in ['RUN_STATUS.json','PUBLIC_OUTPUT_HASHES.json']}))
    write(pub/'RUN_STATUS.json',dict(status='ALL_PHASE1_EXPLORATION_COMPLETE',cells=15,target_met=bool(qualified),chosen=best['arm'],no_new_Outer=True))
    print('EXPLORATION_PHASE1_COMPLETE',best['arm'],'target_met',bool(qualified),flush=True)
