"""Aggregate frozen development outcomes, cluster bootstrap and residual audits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import sha,json_write,patient_metrics,METRICS,state_hash
from prepare import binding


def aggregate(frame,by):
    metrics=frame.groupby(by,sort=True)[METRICS].mean().reset_index()
    counts=frame.groupby(by).agg(patient_fold_cells=('patient','size'),unique_patients=('patient','nunique')).reset_index()
    return metrics.merge(counts,on=by,validate='one_to_one')


def bootstrap(patients):
    ids=sorted(patients.patient.unique()); draws=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    rows=[]
    for left,right in [('T1','T0'),('T2','T0'),('T2','T1')]:
        pair=patients[patients.arm==left].merge(patients[patients.arm==right],on=['patient','fold','center'],suffixes=('_l','_r'),validate='one_to_one')
        assert len(pair)==65
        for metric in METRICS:
            diff=pair[metric+'_l'].to_numpy()-pair[metric+'_r'].to_numpy()
            temp=pd.DataFrame({'patient':pair.patient,'difference':diff}).groupby('patient').difference.agg(['sum','count']).reindex(ids)
            samples=temp['sum'].to_numpy()[draws].sum(1)/temp['count'].to_numpy()[draws].sum(1)
            lo,hi=np.quantile(samples,[.025,.975])
            rows.append({'comparison':left+'-'+right,'metric':metric,'delta':float(np.nanmean(diff)),
                'ci_low':float(lo),'ci_high':float(hi),'bootstrap_fraction_positive':float((samples>0).mean()),
                'draws':10000,'seed':42,'cluster_patients':47,'paired_patient_fold_cells':65})
    return pd.DataFrame(rows)


def verify(root,protocol):
    pub=root/'public'; status=json.loads((pub/'RUN_STATUS.json').read_text())
    assert status['status']=='ALL_10_TEMPORAL_STUDENTS_COMPLETE' and status['binding']==binding(protocol)
    patients=[]; channels=[]; integrity=[]
    for fold in range(1,6):
        d=torch.load(root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt',weights_only=False)
        va=d['val']; score=torch.sigmoid(torch.from_numpy(d['base_logits'][va])).numpy()
        base=pd.DataFrame({'patient':d['patient'][va],'channel':d['channel'][va],'center':d['center'][va],
            'y_nez':d['y'][va],'score_nez':score,'delta_nez':0.,'base_logit':d['base_logits'][va],
            'threshold':d['T0_threshold'],'T0_threshold':d['T0_threshold'],'fold':fold,'arm':'T0'})
        pm=patient_metrics(base.y_nez.to_numpy(),score,base.patient.to_numpy(),base.center.to_numpy(),d['T0_threshold'])
        pm['arm']='T0'; pm['fold']=fold; patients.append(pm); channels.append(base)
        initial=None
        for arm in ['T1','T2']:
            cell=root/f'fold{fold}'/arm; meta=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            assert sha(cell/'LAST_PRIVATE.pt')==meta['last_sha256'] and sha(cell/'BEST_PRIVATE.pt')==meta['best_sha256']
            last=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            best=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert last['binding']==best['binding']==meta['binding']
            assert state_hash(last['best']['model'])==state_hash(best['model'])
            assert len(last['history'])==last['epoch']+1 and 'rng' in last and 'optimizer' in last
            initial=initial or meta['binding']['initial']; assert meta['binding']['initial']==initial
            c=pd.read_csv(cell/'PREDICTIONS_PRIVATE.csv'); assert len(c)==len(base)
            pair=c.merge(base,on=['patient','channel','center','y_nez','fold'],validate='one_to_one',suffixes=('_new','_base'))
            assert len(pair)==len(base) and np.isfinite(c.score_nez).all()
            assert np.max(np.abs(c.delta_nez))<=.5+1e-7 and np.all(c.threshold==meta['threshold'])
            m=patient_metrics(c.y_nez.to_numpy(),c.score_nez.to_numpy(),c.patient.to_numpy(),c.center.to_numpy(),meta['threshold'])
            old=pd.read_csv(cell/'PATIENT_PRIVATE.csv')
            join=m.merge(old,on=['patient','center'],validate='one_to_one',suffixes=('_n','_o'))
            assert all(np.allclose(join[k+'_n'],join[k+'_o'],rtol=0,atol=1e-12) for k in METRICS)
            assert float(m.macro_f1.mean())>=float(pm.macro_f1.mean())-1e-12
            m['arm']=arm; m['fold']=fold; patients.append(m); channels.append(c)
            integrity.append({'fold':fold,'arm':arm,'selected_epoch':meta['selected_epoch'],
                'threshold':meta['threshold'],'completed_epochs':last['epoch'],'checkpoint_sha256':meta['best_sha256'],
                'private_prediction_sha256':sha(cell/'PREDICTIONS_PRIVATE.csv'),'initial_sha256':initial})
    patients=pd.concat(patients,ignore_index=True); channels=pd.concat(channels,ignore_index=True)
    assert patients.patient.nunique()==47 and patients.groupby('arm').size().eq(65).all()
    json_write(pub/'ARTIFACT_INTEGRITY_AUDIT.json',{'status':'PASS','cells':integrity,
        'selected_artifact_metrics_hashes_verified':True,'epoch0_noninferiority_verified':True,'outer_evaluation':False})
    return patients,channels


def diagnostic(channels,pub):
    decisions=[]; residuals=[]; fixed=[]
    base=channels[channels.arm=='T0']
    for arm in ['T1','T2']:
        c=channels[channels.arm==arm]
        pair=c.merge(base,on=['patient','channel','center','fold','y_nez'],suffixes=('_new','_base'),validate='one_to_one')
        for name,groups in [('overall',[('ALL',pair)]),('fold',list(pair.groupby('fold'))),('center',list(pair.groupby('center')))]:
            for unit,g in groups:
                y=g.y_nez.to_numpy(); original=g.score_nez_base.to_numpy()>=g.threshold_base.to_numpy()
                for mode,tau in [('own_selected',g.threshold_new.to_numpy()),('fixed_T0',g.threshold_base.to_numpy())]:
                    prediction=g.score_nez_new.to_numpy()>=tau
                    correct_base=original==y; correct_new=prediction==y
                    base_tp=(y==0)&~original; new_tp=(y==0)&~prediction
                    base_fp=(y==1)&~original; new_fp=(y==1)&~prediction
                    decisions.append({'arm':arm,'scope':name,'unit':str(unit),'threshold':mode,'channels':len(g),
                        'corrected':int((~correct_base&correct_new).sum()),'spoiled':int((correct_base&~correct_new).sum()),
                        'net_corrected':int(correct_new.sum()-correct_base.sum()),
                        'EZ_TP_gained':int((new_tp&~base_tp).sum()),'EZ_TP_lost':int((base_tp&~new_tp).sum()),
                        'FP_added':int((new_fp&~base_fp).sum()),'FP_removed':int((base_fp&~new_fp).sum())})
        for (fold,pid),g in pair.groupby(['fold','patient']):
            delta=g.delta_nez_new.to_numpy(); logit=g.base_logit_new.to_numpy()
            old=g.score_nez_base.to_numpy(); new=g.score_nez_new.to_numpy()
            i,j=np.triu_indices(len(g),1); original=np.sign(old[i]-old[j]); after=np.sign(new[i]-new[j])
            sd=delta.std(); mean=delta.mean(); rms=np.sqrt(np.mean(delta**2))
            residuals.append({'arm':arm,'fold':fold,'patient':pid,'center':str(g.center.iloc[0]),
                'residual_mean':float(mean),'within_patient_residual_std':float(sd),
                'mean_absolute_residual':float(np.abs(delta).mean()),'max_absolute_residual':float(np.abs(delta).max()),
                'saturation_fraction':float((np.abs(delta)>=.49).mean()),
                'constant_shift_energy_fraction':float(mean**2/(rms**2)) if rms>0 else 0.,
                'correlation_with_A0_logit':float(np.corrcoef(delta,logit)[0,1]) if sd>1e-10 and logit.std()>0 else np.nan,
                'pair_order_relation_changed_fraction':float((original!=after).mean()),
                'strict_pair_rank_reversal_fraction':float(((original*after)<0).mean())})
        for fold,g in c.groupby('fold'):
            m=patient_metrics(g.y_nez.to_numpy(),g.score_nez.to_numpy(),g.patient.to_numpy(),g.center.to_numpy(),float(g.T0_threshold.iloc[0]))
            m['arm']=arm; m['fold']=fold; fixed.append(m)
    pd.DataFrame(decisions).to_csv(pub/'DECISION_CHANGE_AUDIT.csv',index=False)
    rf=pd.DataFrame(residuals); values=[k for k in rf.columns if k not in ['arm','fold','patient','center']]
    allres=rf.groupby('arm')[values].mean().reset_index(); allres['scope']='overall'; allres['unit']='ALL'
    more=[]
    for name in ['fold','center']:
        r=rf.groupby(['arm',name])[values].mean().reset_index().rename(columns={name:'unit'}); r['scope']=name; more.append(r)
    pd.concat([allres]+more,ignore_index=True).to_csv(pub/'RESIDUAL_DIAGNOSTICS.csv',index=False)
    return pd.concat(fixed,ignore_index=True),allres


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2); pub=a.runtime/'public'
    patients,channels=verify(a.runtime,a.protocol)
    summary=aggregate(patients,['arm']); folds=aggregate(patients,['arm','fold']); centers=aggregate(patients,['arm','center'])
    for name,frame in [('VALIDATION_SUMMARY',summary),('VALIDATION_BY_FOLD',folds),('VALIDATION_BY_CENTER',centers),('PAIRED_BOOTSTRAP',bootstrap(patients))]:
        frame.to_csv(pub/(name+'.csv'),index=False)
    fixed,residual=diagnostic(channels,pub)
    aggregate(pd.concat([fixed,patients[patients.arm=='T0']],ignore_index=True),['arm']).to_csv(pub/'FIXED_T0_THRESHOLD_DIAGNOSTIC.csv',index=False)
    s=summary.set_index('arm'); delta=lambda metric,left,right:float(s.loc[left,metric]-s.loc[right,metric])
    f=folds.pivot(index='fold',columns='arm',values='macro_f1')
    improve0=int((f.T2>f.T0).sum()); improve1=int((f.T2>f.T1).sum())
    # Frozen criterion is a fraction of channel appearances, not a mean of
    # patient fractions (the public residual summaries remain patient-equal).
    sat=float((np.abs(channels.loc[channels.arm=='T2','delta_nez'].to_numpy())>=.49).mean())
    conditions={'T2_T0_macro_ge_0_015':delta('macro_f1','T2','T0')>=.015,
        'T2_T1_macro_ge_0_005':delta('macro_f1','T2','T1')>=.005,
        'EZ_AP_nondecline':delta('ez_auprc','T2','T0')>=0,'EZ_F1_nondecline':delta('ez_f1','T2','T0')>=0,
        'four_of_five_T2_T0_improved_folds':improve0>=4,'three_of_five_T2_T1_improved_folds':improve1>=3,
        'mask_leakage_saturation_checks':sat<=.10}
    passed=all(conditions.values()); target=delta('macro_f1','T2','T0')>=.03
    if passed and target:terminal='TEMPORAL_COTAR_TARGET_SUPPORTED'
    elif delta('macro_f1','T1','T0')>0 and delta('macro_f1','T2','T1')<=0:terminal='TEMPORAL_FEATURES_USEFUL_COTAR_NOT_SUPPORTED'
    elif delta('macro_f1','T2','T0')>0 and delta('macro_f1','T2','T1')>0:terminal='TEMPORAL_COTAR_POSITIVE_BELOW_TARGET'
    else:terminal='TEMPORAL_FEATURE_INCREMENT_NOT_SUPPORTED'
    json_write(pub/'VALIDATION_GATE.json',{'pass':passed,'terminal':terminal,'conditions':conditions,
        'failed_conditions':[k for k,v in conditions.items() if not v],'target_gain_ge_0_030':target,
        'T2_T0_macro_delta':delta('macro_f1','T2','T0'),'T2_T1_macro_delta':delta('macro_f1','T2','T1'),
        'T2_T0_AP_delta':delta('ez_auprc','T2','T0'),'T2_T0_EZ_F1_delta':delta('ez_f1','T2','T0'),
        'T2_T0_improved_folds':improve0,'T2_T1_improved_folds':improve1,'severe_saturation':sat>.10,
        'saturated_validation_channel_fraction':sat,
        'no_new_outer_evaluation_authorized':True,'terminal_positive_does_not_imply_statistical_support':True})
    json_write(pub/'RUN_STATUS.json',{'status':'COMPLETE_DEVELOPMENT_GATE_'+('PASS' if passed else 'FAILED'),
        'terminal':terminal,'student_models':10,'T0_retrained':False,'outer_evaluation':False,
        'validation_appearances':65,'unique_validation_patients':47,'bootstrap_draws':10000,
        'protocol_sha256':sha(a.protocol),'binding':binding(a.protocol),'finalizer_sha256':sha(Path(__file__))})
    print(summary.to_string(index=False),flush=True); print('TEMPORAL_AGGREGATION_COMPLETE',terminal,flush=True)


if __name__=='__main__':main()
