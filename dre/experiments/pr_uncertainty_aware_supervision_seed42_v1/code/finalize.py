"""Aggregate-only reporting and patient-cluster paired bootstrap; no optimization."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from uas_core import METRICS, sha, json_write, lambda_epoch


def bootstrap(patients):
    outputs=[]; ids=sorted(patients.patient.unique())
    indices=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    for left,right in [('A1','A0'),('A2','A0'),('A2','A1')]:
        l=patients[patients.arm==left]; r=patients[patients.arm==right]
        pair=l.merge(r,on=['patient','fold','center'],suffixes=('_l','_r'),validate='one_to_one')
        assert len(pair)==len(l)==len(r)
        for metric in METRICS:
            pair['difference']=pair[metric+'_l']-pair[metric+'_r']
            aggregate=pair.groupby('patient').difference.agg(['sum','count']).reindex(ids)
            sums=aggregate['sum'].to_numpy(); counts=aggregate['count'].to_numpy()
            samples=sums[indices].sum(1)/counts[indices].sum(1)
            lo,hi=np.quantile(samples,[.025,.975])
            outputs.append({'comparison':left+'-'+right,'metric':metric,
                'delta':float(pair.difference.mean()),'ci_low':lo,'ci_high':hi,
                'bootstrap_fraction_positive':float((samples>0).mean()),
                'draws':10000,'seed':42,'cluster_patients':len(ids),'paired_patient_fold_cells':len(pair)})
    return pd.DataFrame(outputs)


def aggregate(frame,by):
    result=frame.groupby(by,sort=True)[METRICS].mean().reset_index()
    counts=frame.groupby(by).agg(patient_fold_cells=('patient','size'),unique_patients=('patient','nunique')).reset_index()
    return result.merge(counts,on=by,validate='one_to_one')


def uncertainty_reports(root, output):
    summary=[]; soft=[]
    for fold in range(1,6):
        oof=torch.load(root/f'fold{fold}'/'OOF_FROZEN_PRIVATE.pt',weights_only=False)
        y,q,u=[oof[k] for k in ['y_nez','q','u']]
        correction=.25*u*(q-y)
        mismatch=(q>=.5)!=y
        epsilon_matched=2*float(np.abs(correction).mean())
        for name,mask in [('all',np.ones(len(y),bool)),('observed_EZ',y==0),('observed_NEZ',y==1)]:
            summary.append({'fold':fold,'observed_class':name,'channels':int(mask.sum()),
                'u_mean':float(u[mask].mean()),'u_q10':float(np.quantile(u[mask],.1)),
                'u_median':float(np.median(u[mask])),'u_q90':float(np.quantile(u[mask],.9)),
                'high_u_ge_0_8_fraction':float((u[mask]>=.8).mean()),
                'teacher_observed_label_mismatch_fraction':float(mismatch[mask].mean()),
                'u_mean_when_observed_mismatch':float(u[mask&mismatch].mean()) if (mask&mismatch).any() else None,
                'u_mean_when_observed_match':float(u[mask&~mismatch].mean()) if (mask&~mismatch).any() else None,
                'u_AUROC_for_observed_mismatch':float(roc_auc_score(mismatch[mask],u[mask])) if len(set(mismatch[mask]))==2 else None,
                'MC_variance_mean':float(oof['variance'][mask].mean()),
                'expected_entropy_mean':float(oof['expected_entropy'][mask].mean()),
                'mutual_information_mean':float(oof['mutual_information'][mask].mean()),
                'clinical_label_error_identified':False})
            for method,delta in [('A1_fixed_epsilon_0_10',.1*(.5-y)),('A2_max_lambda_0_25',correction),
                                  ('constant_smoothing_same_mean_movement_TARGET_ONLY',epsilon_matched*(.5-y))]:
                w=max(int((y==0).sum()),1)/max(int((y==1).sum()),1)
                grad=lambda target:q*(1+(w-1)*target)-w*target
                soft.append({'fold':fold,'observed_class':name,'method':method,
                    'channels':int(mask.sum()),'mean_abs_target_movement':float(np.abs(delta[mask]).mean()),
                    'q90_abs_target_movement':float(np.quantile(np.abs(delta[mask]),.9)),
                    'max_abs_target_movement':float(np.abs(delta[mask]).max()),
                    'mean_signed_target_movement':float(delta[mask].mean()),
                    'mean_abs_weighted_BCE_gradient_change_at_OOF_q':float(np.abs(grad(y+delta)-grad(y))[mask].mean()),
                    'constant_epsilon_same_mean_movement':epsilon_matched,
                    'target_only_not_extra_trained_arm':method.startswith('constant_')})
        selected=json.loads((root/f'fold{fold}'/'A2'/'SELECTION.json').read_text())
        soft.append({'fold':fold,'observed_class':'all','method':'A2_at_selected_epoch',
            'channels':len(y),'selected_epoch':selected['epoch'],'lambda_at_selection':lambda_epoch(selected['epoch']),
            'mean_abs_target_movement':float((lambda_epoch(selected['epoch'])*u*np.abs(q-y)).mean())})
    pd.DataFrame(summary).to_csv(output/'UNCERTAINTY_SUMMARY.csv',index=False)
    pd.DataFrame(soft).to_csv(output/'SOFT_LABEL_DIAGNOSTICS.csv',index=False)


def decision_report(channel_rows, output):
    rows=[]
    for arm in ['A1','A2']:
        baseline=channel_rows[channel_rows.arm=='A0']
        compared=channel_rows[channel_rows.arm==arm]
        aligned=baseline.merge(compared,on=['patient','channel','fold','center','y_nez'],
                               validate='one_to_one',suffixes=('_b0','_method'))
        assert len(aligned)==len(baseline)
        distance=np.abs(aligned.score_nez_b0-aligned.threshold_b0)
        aligned['boundary_bin']=pd.cut(distance,[0,.05,.15,1.],include_lowest=True,
                                      labels=['distance_0_to_.05','distance_.05_to_.15','distance_.15_to_1'])
        aligned['old_correct']=(aligned.score_nez_b0>=aligned.threshold_b0)==aligned.y_nez
        aligned['new_correct']=(aligned.score_nez_method>=aligned.threshold_method)==aligned.y_nez
        for level in ['all','boundary_bin','center','fold']:
            pieces=[('all',aligned)] if level=='all' else aligned.groupby(level,observed=True)
            for name,g in pieces:
                old=g.old_correct.to_numpy(); new=g.new_correct.to_numpy()
                # Separate operating point effects using A0's frozen numeric threshold.
                same_tau=(g.score_nez_method>=g.threshold_b0)==g.y_nez
                rows.append({'arm':arm,'level':level,'group':str(name),'channels':len(g),
                    'corrected_B0_errors':int((~old&new).sum()),'spoiled_B0_correct':int((old&~new).sum()),
                    'net_corrected':int(new.sum()-old.sum()),
                    'net_corrected_at_A0_numeric_threshold':int(same_tau.sum()-old.sum()),
                    'mean_abs_score_change':float(np.abs(g.score_nez_method-g.score_nez_b0).mean()),
                    'mean_abs_threshold_change':float(np.abs(g.threshold_method-g.threshold_b0).mean())})
    pd.DataFrame(rows).to_csv(output/'DECISION_CHANGE_AUDIT.csv',index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    root=a.runtime; output=root/'public'; output.mkdir(exist_ok=True)
    state=json.loads((root/'RUN_STATUS.json').read_text())
    assert state['status']=='ALL_DEVELOPMENT_TRAINING_COMPLETE'
    frames=[]; channel_frames=[]; selections=[]
    for fold in range(1,6):
        for arm in ['A0','A1','A2']:
            cell=root/f'fold{fold}'/arm
            selection=json.loads((cell/'SELECTION.json').read_text()); selections.append(selection)
            assert sha(cell/'BEST_PRIVATE.pt')==selection['checkpoint_sha256']
            frames.append(pd.read_csv(cell/'VALIDATION_PATIENT_PRIVATE.csv'))
            channel_frames.append(pd.read_csv(cell/'VALIDATION_CHANNEL_PRIVATE.csv'))
    patients=pd.concat(frames,ignore_index=True); channels=pd.concat(channel_frames,ignore_index=True)
    assert patients.groupby(['arm','fold']).size().eq(13).all()
    for fold in range(1,6):
        assert len({s['initial_hash'] for s in selections if s['fold']==fold})==1
    byfold=aggregate(patients,['arm','fold']); byfold.to_csv(output/'VALIDATION_BY_FOLD.csv',index=False)
    aggregate(patients,['arm','center']).to_csv(output/'VALIDATION_BY_CENTER.csv',index=False)
    aggregate(patients,['arm']).to_csv(output/'VALIDATION_SUMMARY.csv',index=False)
    boot=bootstrap(patients); boot.to_csv(output/'PAIRED_BOOTSTRAP.csv',index=False)
    uncertainty_reports(root,output); decision_report(channels,output)
    summary=aggregate(patients,['arm']).set_index('arm')
    delta=lambda metric,left,right:float(summary.loc[left,metric]-summary.loc[right,metric])
    improving=int(((byfold[byfold.arm=='A2'].set_index('fold').macro_f1-
                    byfold[byfold.arm=='A0'].set_index('fold').macro_f1)>0).sum())
    conditions={'A2_minus_A0_macro_ge_0_015':delta('macro_f1','A2','A0')>=.015,
        'A2_minus_A1_macro_positive':delta('macro_f1','A2','A1')>0,
        'A2_EZ_AP_nondecline':delta('ez_auprc','A2','A0')>=0,
        'A2_EZ_F1_nondecline_conservative_materiality':delta('ez_f1','A2','A0')>=0,
        'four_of_five_folds_improved':improving>=4}
    passed=all(conditions.values())
    if delta('macro_f1','A2','A1')<=0: terminal='A2_NO_INCREMENT_OVER_SMOOTHING'
    elif delta('macro_f1','A2','A0')<=0: terminal='A2_UNCERTAINTY_CORRECTION_NOT_SUPPORTED'
    elif delta('macro_f1','A2','A0')>=.03 and passed: terminal='A2_TARGET_GAIN_SUPPORTED'
    else: terminal='A2_POSITIVE_BUT_BELOW_TARGET'
    gate={'pass':passed,'conditions':conditions,'failed_conditions':[k for k,v in conditions.items() if not v],
          'improved_folds':improving,'delta_macro_A2_A0':delta('macro_f1','A2','A0'),
          'delta_macro_A2_A1':delta('macro_f1','A2','A1'),'delta_AP_A2_A0':delta('ez_auprc','A2','A0'),
          'delta_EZ_F1_A2_A0':delta('ez_f1','A2','A0'),'target_delta_ge_0_030':delta('macro_f1','A2','A0')>=.03,
          'development_terminal':terminal,'outer_test_permitted':passed,'exploratory_repeated_test':True}
    json_write(output/'VALIDATION_GATE.json',gate)
    for name in ['A0_REPRODUCTION.json','OOF_TEACHER_AUDIT.json','TEST_AND_SMOKE_AUDIT.json']:
        source=root/'gate'/name if name=='A0_REPRODUCTION.json' else root/name
        json_write(output/name,json.loads(source.read_text()))
    json_write(output/'SELECTION_AUDIT.json',{'selections':[{k:v for k,v in s.items() if k!='metrics'} for s in selections],
        'all_student_initial_states_identical_within_fold':True,
        'A2_selected_before_nonzero_correction_folds':[s['fold'] for s in selections if s['arm']=='A2' and s['epoch']<=5]})
    status={'status':'DEVELOPMENT_GATE_PASS_REQUIRES_OUTER_FREEZE' if passed else 'COMPLETE_DEVELOPMENT_GATE_FAILED',
        'terminal':terminal,'development_complete':True,'outer_evaluation_run':False,
        'historical_B0_only_replay_run':True,'validation_appearances_per_arm':len(patients)//3,
        'unique_validation_patients':int(patients.patient.nunique()),
        'protocol_sha256':sha(a.protocol),'model_cells':15,'teachers':20}
    json_write(output/'RUN_STATUS.json',status)
    print(json.dumps({'summary':summary[METRICS].to_dict(orient='index'),'gate':gate,'status':status},indent=2),flush=True)


if __name__=='__main__': main()
