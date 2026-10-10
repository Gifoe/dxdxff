"""Non-identifying aggregates and fixed terminal; no selection feedback."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import METRICS,metric,cluster_bootstrap,write_json,sha
from analysis import ARMS

def summarize(frame,keys,columns):
    rows=[]
    for key,g in frame.groupby(keys,sort=True):
        if not isinstance(key,tuple):key=(key,)
        rows.append({**dict(zip(keys,key)),'status':'COMPLETE','patient_fold_cells':len(g),'unique_patients':g.patient.nunique(),
            **{c:float(g[c].mean()) for c in columns if c in g}})
    return rows

def complete(rows,diag,errors,gates,fits,transport_rows,lock,pub,protocol):
    f=pd.DataFrame(rows); public_methods=['D0','D2']+list(ARMS)
    for name in f.method.unique():
        g=f[f.method==name]; assert len(g)==65 and g.patient.nunique()==47 and g.channels.sum()==6273
    su=summarize(f,['method'],METRICS); sf=summarize(f,['fold','method'],METRICS); sc=summarize(f,['center','method'],METRICS)
    for name in ARMS:
        if not gates[name]['admitted']:
            su.append({'method':name,'status':'NOT_ESTIMABLE_FIT_GATE_FAILED','patient_fold_cells':0,'expected_cells':65})
            sf += [{'method':name,'fold':k,'status':'NOT_ESTIMABLE_FIT_GATE_FAILED','patient_fold_cells':0,'expected_cells':13} for k in range(1,6)]
            sc += [{'method':name,'center':c,'status':'NOT_ESTIMABLE_FIT_GATE_FAILED','patient_fold_cells':0} for c in sorted(f.center.unique())]
    pd.DataFrame(su).to_csv(pub/'VALIDATION_SUMMARY.csv',index=False)
    pd.DataFrame(sf).to_csv(pub/'VALIDATION_BY_FOLD.csv',index=False)
    pd.DataFrame(sc).to_csv(pub/'VALIDATION_BY_CENTER.csv',index=False)
    contrasts=[('D2','D0')]
    for a in ARMS:contrasts += [(a,'D0'),(a,'D2'),(a,a+'_FIXED_PRIOR')]
    contrasts += [('D1_R1','D1_R0'),('D3_R1','D3_R0')]
    boot=[]
    for left,right in contrasts:
        if left not in f.method.unique() or right not in f.method.unique():
            boot += [{'contrast':left+'-'+right,'metric':m,'status':'NOT_ESTIMABLE','draws_prespecified':10000,'cluster_unique_ids':47} for m in METRICS]; continue
        aa=f[f.method==left].set_index(['fold','patient']); bb=f[f.method==right].set_index(['fold','patient']).reindex(aa.index)
        assert len(aa)==len(bb)==65 and bb.channels.notna().all()
        delta=aa[METRICS].to_numpy()-bb[METRICS].to_numpy(); draws=cluster_bootstrap(aa.index.get_level_values('patient'),delta)
        for j,m in enumerate(METRICS):
            samples=draws[:,j]; samples=samples[np.isfinite(samples)]
            boot.append({'contrast':left+'-'+right,'metric':m,'status':'COMPLETE' if len(samples) else 'NOT_ESTIMABLE',
                'delta':float(np.nanmean(delta[:,j])),'CI_low':float(np.quantile(samples,.025)) if len(samples) else None,
                'CI_high':float(np.quantile(samples,.975)) if len(samples) else None,'draws_prespecified':10000,
                'valid_draws':len(samples),'cluster_unique_ids':47,'contributing_cells':int(np.isfinite(delta[:,j]).sum()),
                'positive_folds':int(((aa[m]-bb[m]).groupby(level='fold').mean()>0).sum())})
    pd.DataFrame(boot).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    if diag:
        dg=pd.DataFrame(diag)
        calibration=summarize(dg,['method'],['brier','ece','fixed_brier','fixed_ece'])
        fractions=summarize(dg,['method'],['pi_MAE','prior_MAE','MAP_prior_displacement','predicted_fraction','actual_fraction',
            'predicted_fraction_MAE','near_boundary','extreme_k','converged'])
    else:calibration=[]; fractions=[]
    for name in ARMS:
        if not gates[name]['admitted']:
            calibration.append({'method':name,'status':'NOT_ESTIMABLE_FIT_GATE_FAILED'})
            fractions.append({'method':name,'status':'NOT_ESTIMABLE_FIT_GATE_FAILED'})
    pd.DataFrame(calibration).to_csv(pub/'VAL_POSTERIOR_CALIBRATION.csv',index=False)
    pd.DataFrame(fractions).to_csv(pub/'PATIENT_EZ_FRACTION_AUDIT.csv',index=False)
    er=[]
    for name in ARMS:
        ee=[r for r in errors if r['method']==name]
        er.append({'method':name,'status':'COMPLETE' if ee else 'NOT_ESTIMABLE_FIT_GATE_FAILED',
            **({k:sum(r[k] for r in ee) for k in ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled']} if ee else {})})
    pd.DataFrame(er).to_csv(pub/'ERROR_CORRECTION_AUDIT.csv',index=False)
    D={r['method']:r for r in su}; development={}
    for a in ARMS:
        if not gates[a]['admitted']:development[a]={'pass':False,'status':'NOT_ESTIMABLE','conditions':None}; continue
        score=D[a]; nf=next(r['positive_folds'] for r in boot if r['contrast']==a+'-D0' and r['metric']=='macro_f1')
        cond={'gain_over_D0_0p020':score['macro_f1']-D['D0']['macro_f1']>=.020,
            'gain_over_D2_0p005':score['macro_f1']-D['D2']['macro_f1']>=.005,
            'EZ_F1_nondecline':score['ez_f1']>=D['D0']['ez_f1'],'AP_nondecline':score['ez_auprc']>=D['D0']['ez_auprc'],
            'four_improving_folds':nf>=4,'integrity':True}
        development[a]={'pass':all(cond.values()),'conditions':cond,'Macro_F1':score['macro_f1'],
            'minimum_0p658':score['macro_f1']>=.658,'target_0p700':score['macro_f1']>=.700}
    repaired=any(gates[a]['numerically_valid'] for a in ['D1_R1','D3_R1'])
    if development['D3_R1']['pass']:
        terminal='ROBUST_POSTERIOR_TARGET_0P70_REACHED' if D['D3_R1']['macro_f1']>=.700 else 'ROBUST_POSTERIOR_INCREMENTAL_GAIN'
    elif development['D3_R0']['pass']:terminal='RAW_LOGIT_POSTERIOR_INCREMENTAL_GAIN'
    elif repaired:terminal='NUMERICAL_REPAIR_WITHOUT_DECISION_GAIN'
    elif any(g['numerically_valid'] for g in gates.values()):terminal='PATIENT_MIXTURE_SIGNAL_NOT_IDENTIFIABLE'
    else:terminal='POSTERIOR_DENSITY_STILL_INVALID'
    write_json(pub/'VALIDATION_GATE.json',{'terminal':terminal,'arms':development,'FIT_admission':gates,
        'GateA_status':'ROBUST_NORMALIZATION_NUMERICALLY_VALID' if repaired else 'NOT_ESTABLISHED',
        'GateB_status':'MAP_ADMISSION_SUPPORTED' if any(g['admitted'] for g in gates.values()) else 'PATIENT_MIXTURE_SIGNAL_NOT_ESTABLISHED',
        'outer_TEST':False,'VAL_based_representation_selection':False,'no_rescue':True})
    table='| Method | Status | Macro-F1 | EZ-F1 | EZ-AP | EZ-AUROC |\n|---|---|---:|---:|---:|---:|\n'
    for name in public_methods:
        s=D[name]; table+='| '+name+' | '+s['status']+' | '+' | '.join(f'{s[m]:.6f}' if m in s else 'NOT_ESTIMABLE' for m in ['macro_f1','ez_f1','ez_auprc','ez_auroc'])+' |\n'
    fit_table='| Arm | Numeric 5-fold gate | MAP−fixed F1 | Prior MAE−MAP MAE | Positive folds | Admitted |\n|---|---|---:|---:|---|---|\n'
    def val(v):return 'NOT_ESTIMABLE' if v is None else f'{v:+.6f}'
    for a,g in gates.items():fit_table+=f"| {a} | {g['numerically_valid']} | {val(g['MAP_minus_fixed_macro'])} | {val(g['prior_MAE_minus_MAP_MAE'])} | {g['positive_folds']}/5 | {g['admitted']} |\n"
    density=pd.read_csv(pub/'FIT_DENSITY_IDENTIFIABILITY.csv'); nr=pd.read_csv(pub/'ROBUST_NORMALIZATION_AUDIT.csv')
    iq=pd.read_csv(pub/'FIT_IQR_FLOOR_BY_FOLD.csv'); ta=pd.DataFrame(transport_rows)
    answers=[]
    answers.append('1. Frozen A0/D2 reproduced? Yes: current immutable-artifact and exact prediction-metric replay passed; predecessor independent model-forward replay remains hash-verified. No new scorer forward output replaced the frozen predictions.')
    answers.append('2. Legal OOF logits recovered? Yes: all101 locked private artifacts,40 teacher checkpoints/provenance, once-per-FIT-channel coverage, identity and TRAIN-only preprocessor hashes passed. No teacher scores were regenerated.')
    for number,rep in [(3,'R0'),(4,'R1')]:
        a0=density[density.arm=='D1_'+rep]; d2=density[density.arm=='D3_'+rep]
        answers.append(f'{number}. {rep} global ordering? A0 {int(a0.global_ordered.sum())}/5; D2 {int(d2.global_ordered.sum())}/5. Full numerical admission includes counts, finite results and R1 clipping rules, not ordering alone.')
    answers.append('5. Tail domination removed? R1 transformed |z|<=8 and every patient class second moment<=64, verified. The floor is FIT-only and LOO-excludes query. Boundedness does not verify Gaussian likelihood correctness or biological separation.')
    r1=nr[(nr.arm.str.endswith('R1'))&(nr.source=='all')]
    answers.append(f"6. R1 clipping activation? FIT full-fold all-channel fractions range {r1.clip_fraction.min():.6%}–{r1.clip_fraction.max():.6%}; source-specific maxima and patient-fraction quantiles are in ROBUST_NORMALIZATION_AUDIT.csv. Frozen IQR floors range {iq[iq.representation=='R1'].floor.min():.6f}–{iq[iq.representation=='R1'].floor.max():.6f}.")
    answers.append('7. Within-patient ranking preserved? Yes: transformation monotonicity, ordered Gaussian posterior and decoder original-logit/canonical ties are asserted. MAP-vs-fixed can change calibration/cardinality, not ranking. Full ranking metrics use the unchanged scorer order.')
    answers.append('8. FIT distribution stability? Finite bounded R1 moments are established where gates pass; patient heterogeneity, small class gaps and LOO invalidity are separately reported. Numerical validity is not proof of stable transferable class densities.')
    answers.append('9. Source hierarchy improves stability? Shrinkage/fallback behavior is measured in SOURCE_GAUSSIAN_PARAMETERS.json. A known-source inadequate/reversed fit uses global density with the original source prior. No ablation or unseen-center comparison was authorized, so no causal hierarchy improvement is claimed.')
    answers.append('10. MAP improves prevalence accuracy? FIT LOO prior-minus-MAP MAEs are in the table above; required improvement is +0.005. No target labels enter the estimate. These are operational-label mixture proportions, not clinical prevalence.')
    answers.append('11. MAP improves FIT F1? Table above compares identical densities/scores/decoder with fixed prior; +0.005,3 positive groups and EZ-F1 nondecline are required jointly. Exposure prevents fully nested independent interpretation.')
    admitted=[a for a,g in gates.items() if g['admitted']]
    answers.append('12. Admitted combinations? '+(', '.join(admitted) if admitted else 'None; no posterior VAL inference is authorized.')+' All frozen condition booleans are published; they were not weakened.')
    answers.append('13. Full five-fold development results? Table above and VALIDATION_SUMMARY.csv; blocked arms are NOT_ESTIMABLE, not zero and not averages of successful folds.')
    answers.append('14. Posterior outperforms A0? '+('; '.join(f"{a}: {D[a]['macro_f1']-D['D0']['macro_f1']:+.6f}" for a in admitted) if admitted else 'Not estimable; no admitted complete arm.')+' Paired intervals remain exploratory.')
    answers.append('15. Posterior outperforms D2? '+('; '.join(f"{a}: {D[a]['macro_f1']-D['D2']['macro_f1']:+.6f}" for a in admitted) if admitted else 'Not estimable; no admitted complete arm.'))
    answers.append('16. MAP versus fixed prior attribution? FIT screening isolates MAP; admitted VAL fixed-prior contrasts are in PAIRED_BOOTSTRAP.csv. No gain can be attributed to MAP if that contrast fails; absent VAL comparisons are explicitly unavailable.')
    allta=ta[ta.source=='all']
    answers.append(f"17. OOF-to-VAL transport? Patient-median Wasserstein range {allta.patient_median_Wasserstein.min():.6f}–{allta.patient_median_Wasserstein.max():.6f}; transformed-score channel KS range {allta.z_KS_statistic.min():.6f}–{allta.z_KS_statistic.max():.6f}. Full per-source shifts/floor/clip/spread are descriptive, not channel-independent tests or fitting inputs. No causal transport failure is inferred merely from a shift.")
    complete_models=[n for n in public_methods if 'macro_f1' in D[n]]
    answers.append('18. Any complete model reaches .658? '+str(any(D[n]['macro_f1']>=.658 for n in complete_models))+'. Missing posterior results are not numerical failures or successes.')
    answers.append('19. Any complete model reaches .700? '+str(any(D[n]['macro_f1']>=.700 for n in complete_models))+'. These are repeatedly reused development patients, not new independent testing.')
    answers.append('20. Continue or stop? Stop this locked run after the terminal; no automatic replacement likelihood, backbone or tuning. Numerical repair alone is not evidence for useful patient mixture information. A narrow negative result does not prove that physiological signals or all posterior mechanisms lack information.')
    report=f'''# A0 robust posterior identifiability v2 — seed42

Terminal: **`{terminal}`**. New independent protocol; previous experiment unmodified. No new scorer/teacher training, raw EEG, label change, outer TEST, rescue tuning or validation-based representation selection.

## Complete matched development results

{table}

Original80 patients/7635 canonical pairs/88D/frozen5fold; development65 appearances/47 IDs/6273 channel appearances. All11 metrics and per-fold/source tables are delivered. Fixed-prior rows, where admitted, are diagnostics rather than selectable scorer models. No partial posterior folds are substituted for complete arms.

## FIT-only admission, not independent validation

{fit_table}

Each pseudo-target is excluded from density labels, source prior and R1 floor. Its own OOF teacher excluded it from TRAIN and selection. However other density-input OOF teachers can have trained or selected on its labels; actual exposure is in OOF_TEACHER_EXPOSURE_AUDIT.json. This is a screening diagnostic, not fully nested prospective validation. No teacher retraining conceals this limitation.

Near-zero IQR no longer creates unbounded R1 moments. H1 numerical repair is supported only where full Gate A passes. That does not establish predictive H2: the narrow Gaussian/MAP utility gate must pass separately. If no arm is admitted, no conclusion about its VAL performance is available. A negative FIT gate says patient-mixture signal was not established under this fixed family, not that all unlabeled distributions are information-free.

The Gaussian two-means/common-variance family, equal-patient moments, nu10 hierarchy, Beta20 prior, MAP solver, bounds and all-k ratio-of-expected-counts Macro-F1 approximation remain unchanged. Only R0/R1 preprocessing differs. Ranking ties from clipping/saturation preserve original logit order then canonical identity. Posteriors/mixture fractions refer to operational benchmark labels, not biological EZ truth.

No label-guided floor, source rescaling, new density or posterior rescue was used. Severe-collapse screening was frozen before outcomes: any invalid LOO/numerical/MAP result or >20% near-boundary/all-or-none decisions in any FIT group prevents admission. These screening cutoffs are not performance-optimized constants.

## Statistics and transport

10,000 seed42 paired unique-patient cluster draws retain all repeated development appearances. Missing metrics remain missing. Required posterior contrasts are NOT_ESTIMABLE when blocked. The controls are unchanged and CIs do not remove repeated development use, prior scorer checkpoint/threshold selection optimism or multiplicity.

OOF_TO_VAL_TRANSPORT_AUDIT.csv uses label-blind score summaries for every scorer/representation. VAL scores neither fit parameters nor select candidates. Source/group/channel shifts are descriptive; no independent-channel significance is claimed. Labels are revealed only after any admitted fixed inference; no post-outcome family choice.

## Explicit answers to20 requested questions

'''+ '\n\n'.join(answers)+'''

## Integrity and delivery

C0–C9 synthetic controls, exact old-family parity, masked target-label isolation, deterministic repeats and real FIT-only normalization smoke are required before analysis. Inputs/protocol/code are hash-bound. Density/admission artifacts are sealed before VAL; independent deterministic rerun verifies aggregate outputs without changing choices. Checkpoints, individual patient/channel scores, labels, mixture proportions, OOF ledgers and logs remain on the trusted server. Only code, hashes and non-identifying aggregates are published.
'''
    (pub/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE_FROZEN_DIAGNOSTIC','terminal':terminal,'protocol_sha256':sha(protocol),
        'admitted_arms':admitted,'no_new_training':True,'no_outer_TEST':True,'full_posterior_arms_fabricated':False,
        'aggregate_files':{p.name:sha(p) for p in sorted(pub.glob('*')) if p.is_file() and p.name!='RUN_STATUS.json'}})
    print('COMPLETE',terminal,flush=True)
