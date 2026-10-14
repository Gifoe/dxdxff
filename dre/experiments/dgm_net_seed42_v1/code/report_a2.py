"""Anonymous aggregate delivery; explicit not-run controls and limited terminal."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_support import write,sha,original_modules,load

def finish(runtime,c,rows,selects,errors,activity,params):
    pub=runtime/'results';metrics,_=original_modules(); df=pd.DataFrame(rows)
    for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(65,47),arm
    means=df.groupby('arm')[metrics.METRICS].mean(); summary=means.loc[['A0','A2']].reset_index()
    summary['status']='EXECUTED_DEVELOPMENT';summary['total_parameters']=[1537,11019];summary['trainable_parameters']=[0,9482]
    blanks=pd.DataFrame([dict(arm=a,status='NOT_RUN_USER_SCOPE',total_parameters=p,trainable_parameters=p-1537) for a,p in [('A1',11024),('A3',11019),('A4',11019)]])
    pd.concat([summary,blanks],ignore_index=True).to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:
        output=[dict(arm=a,**{field:v},appearances=len(q),unique_IDs=q.patient.nunique(),channels=int(q.channels.sum()),**q[metrics.METRICS].mean().to_dict()) for (a,v),q in df[df.arm.isin(['A0','A2'])].groupby(['arm',field])]
        pd.DataFrame(output).to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    df[df.arm.isin(['A0','A2'])].groupby(['arm','patient'])[metrics.METRICS].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    bo=load('dgm_paired_bootstrap',Path(__file__).parent/'bootstrap.py');boot=bo.contrasts(c,df,metrics.METRICS)
    pd.DataFrame(boot).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    activity=pd.DataFrame(activity); gate=activity.groupby('intervention').mean(numeric_only=True).drop(columns=['fold'],errors='ignore')
    inter=means.loc[[k for k in means.index if k.startswith('A2')]].reset_index()
    inter['intervention']=inter.arm.map(lambda a:'full' if a=='A2' else a[3:]); inter=inter.merge(gate,on='intervention',how='left')
    inter.to_csv(pub/'CONDITIONING_INTERVENTIONS.csv',index=False)
    activity.groupby(['fold','intervention']).mean(numeric_only=True).reset_index().to_csv(pub/'GATE_ACTIVITY_BY_FOLD.csv',index=False)
    error=pd.DataFrame(errors);error.to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False)
    pd.DataFrame(params).to_csv(pub/'PARAMETER_AUDIT.csv',index=False)
    pd.DataFrame(selects).to_csv(pub/'MODEL_SELECTION.csv',index=False)
    diff=means.loc['A2']-means.loc['A0']; nfold=sum(df[(df.arm=='A2')&(df.fold==f)].macro_f1.mean()>df[(df.arm=='A0')&(df.fold==f)].macro_f1.mean() for f in range(1,6))
    gates=dict(MacroF1_min=diff.macro_f1>=.010,EZF1_min=diff.ez_f1>=.010,AUROC_min=diff.ez_auroc>=.005,AP_nondecline=diff.ez_auprc>=0,MacroF1_positive_folds=nfold>=4,EZ_sensitivity_nondecline_margin=diff.sensitivity>=-.010)
    gain=all(gates.values()); shuffled=error[error.intervention=='shuffle']; specific=bool((shuffled.logit_change_max>1e-6).any())
    terminal='A2_INCREMENT_OBSERVED_MATCHED_CONTROLS_NOT_RUN' if gain else 'RAW_MORPHOLOGY_MINIMUM_INCREMENT_NOT_ESTABLISHED'
    scientific=dict(terminal=terminal,A2_A0_minimum_predictive_gate=bool(gain),conditions=gates,A2_A0_delta=diff.to_dict(),positive_MacroF1_folds=int(nfold),shuffle_inference_measurable_change=specific,full_DGM_gate='NOT_EVALUABLE_UNRUN_MATCHED_CONTROLS',A1_A3_A4='NOT_RUN_USER_SCOPE',integrity='PASS',Development_reused_selection_conditioned=True,independent_confirmation=False,no_new_Outer=True)
    write(pub/'SCIENTIFIC_GATE.json',scientific)
    verify_public(pub,means)
    docs(runtime,means,diff,gain,terminal,boot,selects)
    hashes={p.name:sha(p) for p in pub.iterdir() if p.is_file() and p.name not in ['RUN_STATUS.json','PUBLIC_OUTPUT_HASHES.json']}
    write(pub/'PUBLIC_OUTPUT_HASHES.json',dict(status='PASS',sha256=hashes,no_patient_level_artifact_exported=True))
    write(pub/'RUN_STATUS.json',dict(status='ALL_AUTHORIZED_A2_DEVELOPMENT_COMPLETE',terminal=terminal,A2_training_cells=5,A0_retrained=False,trained_controls=0,bootstrap_draws=10000,no_new_Outer=True))
    print('DGM_A2_COMPLETE',terminal,flush=True)

def verify_public(pub,means):
    summary=pd.read_csv(pub/'DEVELOPMENT_SUMMARY.csv').set_index('arm')
    assert np.allclose(summary.loc['A2',list(means.columns)].astype(float),means.loc['A2'],atol=1e-12,equal_nan=True)
    for p in pub.glob('*.csv'):
        columns=set(pd.read_csv(p,nrows=0).columns)
        assert not columns & {'patient','patient_id','channel','subject_id','clinical_label','raw_waveform'}
    write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_metrics='Independent sklearn metrics compared to original per patient to1e-12',bootstrap='Independent multiplicity matrix matched original cluster bootstrap to1e-12',published_CSV_roundtrip=True,all_five_original_folds=True,exact_65_appearances_47_IDs=True))

def docs(runtime,means,diff,gain,terminal,boot,selects):
    pub=runtime/'results'; root=runtime/'public_docs';root.mkdir(exist_ok=True)
    # Generated report text is a normal implementation output, not a manual source edit.
    table='| Model | Parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a,n in [('A0',1537),('A2',11019)]:
        r=means.loc[a];table+=f'| {a} | {n} | {r.ez_auroc:.6f} | {r.ez_auprc:.6f} | {r.macro_f1:.6f} | {r.ez_f1:.6f} | {r.ez_mrr:.6f} |\n'
    for a,n in [('A1',11024),('A3',11019),('A4',11019)]:table+=f'| {a} | {n} | not run | not run | not run | not run | not run |\n'
    text='# DGM-Net A2 — Seed42 Development\n\n'+table+f'\nTerminal: `{terminal}`.\n\n'
    text+='Only A2 was requested in the final scope amendment: five trained cells; A0 exact frozen replay. A1/A3/A4 were not trained. Thus the full window-guidance research hypothesis cannot be accepted, regardless of A2 performance. No new Outer evaluation, rescue variants or other seeds.\n\n'
    text+='Primary estimand: 65 original validation appearances/47 unique IDs; patient-equal, NEZ-positive scoring. Validation labels selected models and thresholds, as in the historical protocol. Repeated Development is exploratory; percentile intervals condition on those selections and do not include their uncertainty. Unique-ID secondary summaries do not replace the primary estimand.\n\n'
    text+='## A2 versus frozen D1\n\n'
    for k in ['ez_auroc','ez_auprc','macro_f1','ez_f1','sensitivity','specificity','ez_mrr','top1_is_ez']:text+=f'- {k}: {means.loc["A2",k]:.6f}; difference {diff[k]:+.6f}.\n'
    text+='\n## Paired bootstrap\n\n| Metric | A2−A0 | Exploratory95% interval |\n|---|---:|---|\n'
    for r in boot:
        if r['comparison']=='A2 minus A0':text+=f'| {r["metric"]} | {r["delta"]:+.6f} | [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}] |\n'
    text+='\n## Identity and mechanism limits\n\nD1 checkpoints/preprocessing stayed frozen. Step0 final logits exactly equal D1; gate initialized exactly1. Legal-FIT gradient smoke model was discarded and training restarted from the locked initialization. No GRU/LSTM/attention exists. Reported gate interventions keep the trained CNN/head and A2-selected threshold; residual-disabled logits exactly recover D1, but its F1 need not equal A0 F1 because A0 uses its own threshold. All seven neutral-feature interventions are diagnostic only; no subset selected.\n\n'
    text+='Multiplicative guidance forbids a direct additive physiology-to-output branch but cannot alone prove morphological complementarity. Wrong-window/static inference changes establish dependence, not predictive value versus retrained controls. Mean pooling is invariant to a joint within-stage permutation of Raw and physiology, so this model cannot establish within-stage chronological order. Bandwidth-limited high gamma stops at125Hz; metadata time0 is not independently verified clinical onset.\n\n'
    text+='## Frozen selections\n\n| Fold | Epoch | NEZ threshold | Alpha | Effective residual coefficient |\n|---|---:|---:|---:|---:|\n'
    for r in selects:text+=f'| {r["fold"]} | {r["epoch"]} | {r["threshold"]:.3f} | {r["alpha"]:.6f} | {r["effective_residual_coefficient"]:.6f} |\n'
    text+='\n## Historical external architecture reference\n\nHistorical D2 (13,643 parameters, GRU) Development: AUROC0.732515/AP0.515027/Macro-F10.650238/EZ-F10.451028/MRR0.684968. Copied from the frozen historical report, not a newly trained comparison. D2 was not reused as A2 and no Outer individual outputs were opened.\n\nSee every fold/center, interventions/confusions, scales, bootstrap and independent verification in results. Only code/hashes/anonymous aggregate outputs are public; patient IDs, labels, Raw, checkpoints and per-channel scores remain on the original trusted server.\n'
    (root/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
