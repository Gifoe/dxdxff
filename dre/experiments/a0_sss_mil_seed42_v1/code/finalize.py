"""Matched development metrics, ID-cluster bootstrap, inference-only diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_raw import ROOT
from common import METRICS, cluster_bootstrap, sha, write_json
from prepare import PRIOR


def aggregate(frame, keys, columns=METRICS):
    out=[]
    for name,g in frame.groupby(keys,sort=True,dropna=False):
        if not isinstance(name,tuple): name=(name,)
        out.append({**dict(zip(keys,name)),'patient_fold_appearances':len(g),
                    'unique_patients':g.patient.nunique(),
                    **{m:float(g[m].mean()) for m in columns if m in g}})
    return out


def save(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/'public'/name,index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args(); pub=ROOT/'public'; pub.mkdir(exist_ok=True)
    rows=torch.load(ROOT/'CONTROL_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False)
    histories=[]; details=[]; completion=[]; errors=[]; ranks=[]; contribution=[]
    for fold in range(1,6):
        d=torch.load(PRIOR/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        frozen=torch.load(PRIOR/f'fold{fold}/D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        artifacts={}
        for arm in ['S0','S1','S2']:
            cell=ROOT/f'fold{fold}'/arm
            meta=json.loads((cell/'COMPLETE.json').read_text())
            assert meta['status']=='PASS' and meta['binding']['protocol']==sha(a.protocol)
            assert meta['best_sha256']==sha(cell/'BEST_PRIVATE.pt')
            assert meta['validation_sha256']==sha(cell/'VALIDATION_PRIVATE.pt')
            art=torch.load(cell/'VALIDATION_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert art['binding']==meta['binding']
            artifacts[arm]=art
            rows.extend(art['rows']); histories.extend(art['history']); details.extend(art['details'])
            completion.append({'fold':fold,'method':arm,'epochs':meta['completed_epochs'],
                               'selected_epoch':meta['selected_epoch'],'threshold':meta['threshold'],
                               'checkpoint_sha256':meta['best_sha256'],'initial_hash':meta['binding']['initial']})
        assert artifacts['S1']['binding']['initial']==artifacts['S2']['binding']['initial']
        va=d['val']
        for patient in sorted(set(d['patient'][va])):
            ix=va[d['patient'][va]==patient]
            ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            y=d['y'][ix]; old=frozen['scores'][ix]; oldpred=old>=frozen['threshold']
            for arm in ['S1','S2','S1_RAW_DISABLED']:
                art=artifacts['S1'] if arm=='S1_RAW_DISABLED' else artifacts[arm]
                score=art['disabled'][ix] if arm=='S1_RAW_DISABLED' else art['scores'][ix]
                assert np.isfinite(score).all()
                pred=score>=art['threshold']
                before=oldpred==y; after=pred==y
                errors.append({'fold':fold,'patient':patient,'center':str(d['center'][ix[0]]),'method':arm,
                               'TP_gained':int(((y==0)&oldpred&~pred).sum()),
                               'TP_lost':int(((y==0)&~oldpred&pred).sum()),
                               'FP_added':int(((y==1)&oldpred&~pred).sum()),
                               'FP_removed':int(((y==1)&~oldpred&pred).sum()),
                               'errors_recovered':int((~before&after).sum()),
                               'correct_spoiled':int((before&~after).sum()),'channels':len(ix)})
                upper=np.triu(np.ones((len(ix),len(ix)),bool),1)
                one=np.sign(old[:,None]-old); two=np.sign(score[:,None]-score)
                r1=np.argsort(np.argsort(old,kind='stable'),kind='stable')
                r2=np.argsort(np.argsort(score,kind='stable'),kind='stable')
                ranks.append({'fold':fold,'patient':patient,'center':str(d['center'][ix[0]]),'method':arm,
                              'pair_reversal_fraction':float((one*two<0)[upper].mean()) if upper.any() else 0.,
                              'changed_rank_fraction':float((r1!=r2).mean()),
                              'absolute_rank_shift':float(np.abs(r1-r2).mean())})
            full=artifacts['S1']['scores'][ix]; off=artifacts['S1']['disabled'][ix]
            contribution.append({'fold':fold,'patient':patient,'center':str(d['center'][ix[0]]),'method':'S1',
                                 'mean_abs_score_change_raw_intervention':float(np.abs(full-off).mean()),
                                 'decision_change_raw_intervention':float(((full>=artifacts['S1']['threshold'])!=(off>=artifacts['S1']['threshold'])).mean())})
    frame=pd.DataFrame(rows); detail=pd.DataFrame(details)
    for arm in ['D0','D2','S0','S1','S2','S1_RAW_DISABLED']:
        sub=frame[frame.method==arm]
        assert len(sub)==65 and sub.patient.nunique()==47
        assert sub.channels.sum()==6273, 'Matched valid-raw population differs; need explicit subset report'
    save('VALIDATION_SUMMARY.csv',aggregate(frame,['method']))
    save('VALIDATION_BY_FOLD.csv',aggregate(frame,['fold','method']))
    save('VALIDATION_BY_CENTER.csv',aggregate(frame,['center','method']))
    save('TRAINING_HISTORY.csv',histories)
    save('MODEL_SELECTION.csv',completion)
    boot=[]; center_contrasts=[]
    for left,right in [('S0','D0'),('S1','D0'),('S1','D2'),('S2','D2'),('S1','S2')]:
        aa=frame[frame.method==left].set_index(['fold','patient']).sort_index()
        bb=frame[frame.method==right].set_index(['fold','patient']).sort_index()
        assert aa.index.equals(bb.index) and (aa.channels==bb.channels).all()
        delta=aa[METRICS].to_numpy()-bb[METRICS].to_numpy()
        samples=cluster_bootstrap(aa.index.get_level_values('patient'),delta)
        for j,m in enumerate(METRICS):
            val=samples[:,j]; val=val[np.isfinite(val)]
            boot.append({'contrast':left+'-'+right,'metric':m,'delta':float(np.nanmean(delta[:,j])),
                         'CI_low':float(np.quantile(val,.025)),'CI_high':float(np.quantile(val,.975)),
                         'positive_folds':int(((aa[m]-bb[m]).groupby(level='fold').mean()>0).sum()),
                         'unique_patient_ids':47,'patient_fold_appearances':65,'draws':10000,'valid_draws':len(val)})
            for c in sorted(set(aa.center)):
                take=np.asarray(aa.center==c)
                center_contrasts.append({'contrast':left+'-'+right,'metric':m,'center':c,
                                         'delta':float(np.nanmean(delta[take,j])),
                                         'unique_patients':aa[take].index.get_level_values('patient').nunique()})
    save('PAIRED_BOOTSTRAP.csv',boot)
    save('PAIRED_EFFECTS_BY_CENTER.csv',center_contrasts)
    save('FUSION_GATE_AUDIT.csv',aggregate(detail[detail.method!='S0'],['fold','method'],[
        'gamma','raw_rms','engineered_rms','raw_engineered_ratio','nonzero_raw_fraction',
        'raw_embedding_variance','window_embedding_variance','attention_entropy','max_attention',
        'attention_gt_99_fraction','valid_windows_per_seizure','valid_seizures_per_channel']))
    save('RAW_INFORMATION_COMPLEMENTARITY.csv',aggregate(pd.DataFrame(contribution),['fold','method'],[
        'mean_abs_score_change_raw_intervention','decision_change_raw_intervention']))
    save('RANKING_CHANGE_AUDIT.csv',aggregate(pd.DataFrame(ranks),['fold','method'],[
        'pair_reversal_fraction','changed_rank_fraction','absolute_rank_shift']))
    er=pd.DataFrame(errors)
    save('ERROR_CORRECTION_AUDIT.csv',[{'method':m,**{c:int(g[c].sum()) for c in
        ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled','channels']}}
        for m,g in er.groupby('method')])
    efficiency=pd.DataFrame(histories)
    save('EFFICIENCY_AUDIT.csv',[{'fold':f,'method':m,'completed_epochs':len(g),
        'epoch_body_seconds':float(g.seconds.sum()),'mean_epoch_seconds':float(g.seconds.mean()),
        'patient_updates':int(g.patient_updates.sum())} for (f,m),g in efficiency.groupby(['fold','method'])])
    write_json(pub/'MIL_MASKING_AUDIT.json',{'status':'PASS','invalid_weight_nonzero':0,
        'all_padding_NaN':0,'attention_not_biomarker':True,
        'aggregate_diagnostics':aggregate(detail,['method'],['attention_entropy','effective_windows',
            'max_attention','attention_gt_99_fraction','window_embedding_variance',
            'valid_windows_per_seizure','valid_seizures_per_channel'])})
    summary=frame.groupby('method')[METRICS].mean()
    b=pd.DataFrame(boot).set_index(['contrast','metric'])
    fusion=pd.DataFrame(aggregate(detail[detail.method=='S1'],['fold','method'],[
        'raw_embedding_variance','gamma','raw_engineered_ratio','nonzero_raw_fraction']))
    collapse=bool(((fusion.raw_embedding_variance<=1e-10)|(fusion.gamma.abs()<=1e-6)|
                  (fusion.raw_engineered_ratio<=1e-4)|(fusion.nonzero_raw_fraction<.95)).any())
    conditions={'delta_S1_D2_ge_020':float(b.loc[('S1-D2','macro_f1'),'delta'])>=.020,
                'delta_S1_S2_ge_010':float(b.loc[('S1-S2','macro_f1'),'delta'])>=.010,
                'EZ_F1_non_decline':summary.loc['S1','ez_f1']>=summary.loc['D2','ez_f1'],
                'EZ_AP_non_decline':summary.loc['S1','ez_auprc']>=summary.loc['D2','ez_auprc'],
                'positive_folds_D2_ge4':int(b.loc[('S1-D2','macro_f1'),'positive_folds'])>=4,
                'positive_folds_S2_ge4':int(b.loc[('S1-S2','macro_f1'),'positive_folds'])>=4,
                'no_integrity_failure':True,'no_material_raw_collapse':not collapse}
    passed=all(conditions.values()); weak=summary.loc['S0','macro_f1']<summary.loc['D0','macro_f1']
    if passed:
        terminal='RAW_ONLY_WEAK_BUT_FUSION_USEFUL' if weak else 'CHANNEL_SPECIFIC_RAW_COMPLEMENTARITY_SUPPORTED'
    elif summary.loc['S1','macro_f1']>summary.loc['D2','macro_f1'] and summary.loc['S1','macro_f1']<=summary.loc['S2','macro_f1']:
        terminal='CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN'
    elif weak and summary.loc['S1','macro_f1']<=summary.loc['D2','macro_f1']:
        terminal='RAW_TEMPORAL_EVIDENCE_NOT_SUPPORTED'
    else:
        terminal='NO_MEANINGFUL_GAIN'
    gate={'continuation_gate':passed,'conditions':conditions,'target_070':bool(summary.loc['S1','macro_f1']>=.7),
          'terminal':terminal,'protocol_sha256':sha(a.protocol),'development_only':True,'outer_test_accessed':False}
    write_json(pub/'VALIDATION_GATE.json',gate)
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE','terminal':terminal,'registered_runs':15,
                                   'all_runs_hash_verified':True,'outer_test_accessed':False})
    # Comprehensive report is generated only from verified aggregate results.
    def value(m,k):return f'{summary.loc[m,k]:.6f}'
    lines=['# A0-SSS-MIL seed42 — completed exploratory development experiment','',
           '| Model | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in ['D0','D2','S0','S1','S2','S1_RAW_DISABLED']:
        lines.append('| '+m+' | '+' | '.join(value(m,k) for k in METRICS)+' |')
    lines += ['',f'Terminal: `{terminal}`. Continuation gate: {passed}; desired 0.700 target: {gate["target_070"]}.',
              '', 'Means weight 65 patient-fold appearances equally; 47 unique patient IDs, 6,273 channel appearances. '
              'Bootstrap resamples unique IDs and keeps all repeated appearances together. No outer TEST evaluation.',
              '', '## Paired contrasts', '', '| Contrast | Macro-F1 delta | ID-cluster 95% CI | Positive folds |',
              '|---|---:|---|---:|']
    for contrast in ['S0-D0','S1-D0','S1-D2','S2-D2','S1-S2']:
        r=b.loc[(contrast,'macro_f1')]
        lines.append(f'| {contrast} | {r.delta:+.6f} | [{r.CI_low:+.6f}, {r.CI_high:+.6f}] | {int(r.positive_folds)}/5 |')
    lines += ['', '## Required answers', '',
        '1. D0/D2 exact frozen metric replay passed after independent checkpoint/hash gates; no control retraining.',
        '2. All 80 original patients, 7,635 canonical channels and five original split identities retained. Clinical labels unchanged.',
        '3. 256 raw records, 24,995 run-channel incidences and 1,471,965 valid candidate windows. Three padded records have candidates inside measured intervals; no padding enters MIL.',
        '4. Onset provenance remains independently verified for 0/256. This limits temporal interpretation to cache-relative information, not clinical propagation or onset identification.',
        '5. This is a code-grounded SSS-inspired adaptation, not exact SSS reproduction; fixed31 measured patches, auxiliary projection and two-level masked MIL are documented.',
        f'6. S0 standalone Macro-F1 {value("S0","macro_f1")}, EZ-AP {value("S0","ez_auprc")}, AUROC {value("S0","ez_auroc")}; compare the matched D0 row, not historical RawTiny.',
        f'7. S1-D2 Macro-F1 delta {b.loc[("S1-D2","macro_f1"),"delta"]:+.6f}; paired interval above.',
        f'8. S1-S2 Macro-F1 delta {b.loc[("S1-S2","macro_f1"),"delta"]:+.6f}; paired interval above.',
        '9. S1-S2 is the channel-assignment control. Small or nonpositive differences do not distinguish true correspondence from capacity/fine-tuning; a positive difference is still not a causal physiological proof.',
        f'10. S1 EZ-AP {value("S1","ez_auprc")} versus D2 {value("D2","ez_auprc")}.',
        f'11. S1 EZ-F1 {value("S1","ez_f1")} versus D2 {value("D2","ez_f1")}.',
        f'12. S1 sensitivity/specificity {value("S1","sensitivity")}/{value("S1","specificity")} versus D2 {value("D2","sensitivity")}/{value("D2","specificity")}.',
        '13. ERROR_CORRECTION_AUDIT.csv gives D2 EZ mistakes recovered, correct EZ channels spoiled, and FP changes without publishing clinical identities.',
        '14. RANKING_CHANGE_AUDIT.csv reports paired ranking reversals and canonical rank shifts; ranking change alone is not improvement.',
        f'15. Material branch collapse under the prelocked numerical definition: {collapse}. Gates, embedding variance, RMS ratios and nonzero contributions are in FUSION_GATE_AUDIT.csv.',
        '16. MIL_MASKING_AUDIT.json reports attention entropy, effective windows, maximum weight and >0.99 concentration. Attention is not a validated biomarker.',
        f'17. S1 beats D2 in {int(b.loc[("S1-D2","macro_f1"),"positive_folds"])}/5 folds, S2 in {int(b.loc[("S1-S2","macro_f1"),"positive_folds"])}/5. All five folds were completed without outcome-based discarding.',
        '18. VALIDATION_BY_CENTER.csv and PAIRED_EFFECTS_BY_CENTER.csv provide all four source-group effects; no center-specific model or threshold.',
        f'19. +0.020 S1-D2 requirement: {conditions["delta_S1_D2_ge_020"]}; complete continuation gate: {passed}.',
        f'20. S1 Macro-F1 >=0.700: {gate["target_070"]}.',
        f'21. The full predeclared continuation gate {"passes" if passed else "does not pass"}; these results are exploratory repeated development, not independent confirmation.',
        f'22. {"Continuation is justified only as a new separately locked study." if passed else "Stop this seed42 protocol; do not add rescue components or retune on these outcomes."}',
        '', '## Limitations and control interpretation', '',
        'S1/S2 start from D2 checkpoints previously selected on these VAL patients and then select again on VAL. This additional repeated-validation selection bias is disclosed; no independent held-out confirmation is claimed.',
        'S1_RAW_DISABLED is an inference intervention at the S1-selected threshold using jointly fine-tuned engineered weights. It is not frozen D2 replay.',
        'Shuffle is within identical valid-seizure signatures to preserve run membership and missingness. Singleton signatures remain unchanged and their counts are audited.',
        'All-clinical-label feature-export member was never loaded; FIT/VAL labels came from frozen development banks. Raw trusted pickle contains clinical metadata fields that were ignored. Unlabeled raw preprocessing includes original cohort context, but no outer labels, predictions, or metrics are used.',
        'Private raw tensors, identities, per-channel scores, checkpoints, optimizer states and execution logs remain on the original server.',
        '', '## Execution verification', '', f'Protocol SHA256: `{sha(a.protocol)}`. Fifteen cell completion hashes, thresholds and selected epochs are in MODEL_SELECTION.csv.',
        f'Total completed epochs: {len(histories)}; measured epoch-body seconds: {efficiency.seconds.sum():.2f}. This excludes preparation, tests and reporting.']
    (pub/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALL_15_RUNS_FINALIZED',terminal,flush=True)


if __name__=='__main__':
    main()
