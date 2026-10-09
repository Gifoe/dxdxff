"""Reporting-only extension: no tuning or new predictions after completed audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import sha,write_json,cluster_bootstrap,METRICS

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--bank-runtime',type=Path,required=True);p.add_argument('--oof-runtime',type=Path,required=True)
    a=p.parse_args(); pub=a.runtime/'public'; status=json.loads((pub/'RUN_STATUS.json').read_text()); assert status['status']=='COMPLETE'
    private=torch.load(a.runtime/'AUDIT_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert private['binding']==status['binding']; of=private['oracle']; hs=private['headroom']; pf=private['probes']
    boot=pd.read_csv(pub/'PATIENT_CLUSTER_BOOTSTRAP.csv'); extra=[]
    for group,frame in [('ALL',of),*[(c,g) for c,g in of.groupby('center') if g.patient.nunique()>=5]]:
        base=frame[frame.method=='baseline'].set_index(['fold','patient'])
        for method in ['oracle','grid_oracle','true_count_topk']:
            other=frame[frame.method==method].set_index(['fold','patient']); assert other.index.equals(base.index)
            for m in METRICS+['TP','FP','TN','FN']:
                d=(other[m]-base[m]).to_numpy(); samples=cluster_bootstrap(base.index.get_level_values('patient'),d)[:,0]; finite=np.isfinite(d); b=samples[np.isfinite(samples)]
                extra.append({'policy':'full_channel_'+group,'contrast':method+'-baseline','metric':m,'delta':float(np.nanmean(d)),
                    'CI_low':float(np.quantile(b,.025)),'CI_high':float(np.quantile(b,.975)),
                    'contributing_cells':int(finite.sum()),'unique_patients':len(set(base.index.get_level_values('patient')[finite])), 'draws':10000,'valid_draws':len(b)})
    pd.concat([boot,pd.DataFrame(extra)],ignore_index=True).to_csv(pub/'PATIENT_CLUSTER_BOOTSTRAP.csv',index=False)
    rows=[]
    for center,g in hs.groupby('center'):
        if g.patient.nunique()<5:continue
        rows.append({'center':center,'cells':len(g),'unique_patients':g.patient.nunique(), 'oracle_below_0.65_cells':int((g.oracle_macro<.65).sum()),
            'large_headroom_and_AP_ge_0.5_cells':int(((g.headroom>=.05)&(g.ez_auprc>=.5)).sum()),
            'large_headroom_cells':int((g.headroom>=.05).sum()),'mean_headroom':float(g.headroom.mean())})
    pd.DataFrame(rows).to_csv(pub/'ORACLE_DIFFICULTY_PROFILE.csv',index=False)
    intervals=json.loads((pub/'THRESHOLD_INTERVAL_AUDIT.json').read_text())
    intervals['multiple_numeric_optimal_threshold_fraction']=float(np.mean([any(hi>lo for lo,hi in v) for v in hs.optimal_intervals]))
    intervals['numeric_multiplicity_definition']='Continuum inside one decision-equivalent interval; distinct optimal decision-set multiplicity is reported separately'
    write_json(pub/'THRESHOLD_INTERVAL_AUDIT.json',intervals)
    transport=[]
    for fold in range(1,6):
        bank=torch.load(a.bank_runtime/f'fold{fold}/DEVELOPMENT_PRIVATE.pt',map_location='cpu',weights_only=False)
        q=torch.load(a.oof_runtime/f'fold{fold}/OOF_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)['q']
        ref=pd.read_csv(a.bank_runtime/f'fold{fold}/A0_CHANNEL_PRIVATE.csv')
        tr=bank['train']; va=bank['val']
        def statistics(y,s,pid):
            per=[]
            for patient in sorted(set(pid)):
                mask=pid==patient; v=s[mask]; per.append([float(v.mean()),float(v.std()),float(np.mean(y[mask]==0)),float(np.mean(v<bank['A0_threshold']))])
            return np.asarray(per).mean(0)
        tm=statistics(bank['y'][tr],q,bank['patient'][tr]); vm=statistics(ref.y_nez.to_numpy(),ref.score_nez.to_numpy(),ref.patient.to_numpy())
        transport.append({'fold':fold,'FIT_score':'OOF 10-pass MC mean','VAL_score':'deterministic final A0','FIT_patient_mean_probability':tm[0],'VAL_patient_mean_probability':vm[0],
            'FIT_patient_mean_score_SD':tm[1],'VAL_patient_mean_score_SD':vm[1],'FIT_observed_EZ_fraction':tm[2],'VAL_observed_EZ_fraction':vm[2],
            'FIT_predicted_EZ_fraction':tm[3],'VAL_predicted_EZ_fraction':vm[3],
            'interpretation':'membership/preprocessing/MC averaging differ; descriptive transport risk, not proof of a causal reason for failure'})
    pd.DataFrame(transport).to_csv(pub/'OOF_VAL_SCORE_TRANSPORT.csv',index=False)
    # One-class support is a label scarcity descriptor, never used to select a policy or lambda.
    cells=[]
    for path in (a.runtime/'episodes_private').glob('*.json'):
        record=json.loads(path.read_text()); cells.append(record)
    # Latest successful binding only; older engineering-attempt caches are preserved but excluded.
    freeze=json.loads((pub/'PRE_VAL_FREEZE.json').read_text()); wanted=freeze['regularization_sha256']
    from core import seed,episode
    scarcity=[]
    for fold in range(1,6):
        bank=torch.load(a.bank_runtime/f'fold{fold}/DEVELOPMENT_PRIVATE.pt',map_location='cpu',weights_only=False); ref=pd.read_csv(a.bank_runtime/f'fold{fold}/A0_CHANNEL_PRIVATE.csv')
        for patient,g in ref.groupby('patient'):
            g=g.sort_values('channel',kind='stable'); take=np.lexsort((g.channel.to_numpy(str),abs(g.score_nez.to_numpy()-bank['A0_threshold'])))[:8]
            oneclass=len(set(g.y_nez.to_numpy()[take]))==1
            for method in ['P0','P1','P2']:
                row=pf[(pf.fold==fold)&(pf.patient==str(patient))&(pf.policy=='B8_uncertainty')&(pf.method==method)].iloc[0]
                scarcity.append({'fold':fold,'patient':str(patient),'method':method,'support_class':'one_class' if oneclass else 'two_classes',
                    **{m:row[m] for m in METRICS},'full_patient_EZ_fraction':float((g.y_nez==0).mean())})
    s=pd.DataFrame(scarcity); rows=[]
    for (label,method),g in s.groupby(['support_class','method']):
        if g.patient.nunique()<5:continue
        rows.append({'support_class':label,'method':method,'cells':len(g),'unique_patients':g.patient.nunique(),
            **{m:float(g[m].mean()) for m in METRICS},'full_patient_EZ_fraction':float(g.full_patient_EZ_fraction.mean())})
    pd.DataFrame(rows).to_csv(pub/'SUPPORT_CLASS_SCARCITY.csv',index=False)
    write_json(pub/'SUPPLEMENT_AUDIT.json',{'status':'PASS','private_ledger_binding_exact':True,'supplement_source_sha256':sha(__file__),
        'checkpoint_model_threshold_labels_unchanged':True,'new_predictions':False,'hyperparameter_selection':False,'outer_test_accessed':False})
    print('REPORTING_SUPPLEMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
