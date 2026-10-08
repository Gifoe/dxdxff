"""Frozen development aggregation, integrity audit and descriptive comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baseline import METRICS, sha, json_write, patient_metrics, state_hash
from prepare import binding


def aggregate(frame, by):
    values=frame.groupby(by,sort=True)[METRICS].mean().reset_index()
    sizes=frame.groupby(by).agg(patient_fold_cells=('patient','size'),unique_patients=('patient','nunique')).reset_index()
    return values.merge(sizes,on=by,validate='one_to_one')


def bootstrap(patients):
    ids=sorted(patients.patient.unique()); rows=[]
    indices=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    for left,right in [('B1','A0'),('B2','A0'),('B2','B1')]:
        pair=patients[patients.arm==left].merge(patients[patients.arm==right],
            on=['patient','fold','center'],validate='one_to_one',suffixes=('_l','_r'))
        assert len(pair)==65
        for metric in METRICS:
            pair['difference']=pair[metric+'_l']-pair[metric+'_r']
            g=pair.groupby('patient').difference.agg(['sum','count']).reindex(ids)
            draws=g['sum'].to_numpy()[indices].sum(1)/g['count'].to_numpy()[indices].sum(1)
            lo,hi=np.quantile(draws,[.025,.975])
            rows.append({'comparison':left+'-'+right,'metric':metric,'delta':float(pair.difference.mean()),
                'ci_low':lo,'ci_high':hi,'bootstrap_fraction_positive':float((draws>0).mean()),
                'draws':10000,'seed':42,'cluster_patients':len(ids),'paired_patient_fold_cells':len(pair)})
    return pd.DataFrame(rows)


def verify(root, protocol):
    runtime=json.loads((root/'RUN_STATUS.json').read_text())
    assert runtime['status']=='ALL_15_STUDENTS_COMPLETE'
    assert runtime['binding']==binding(protocol)
    assert json.loads((root/'public/A0_REPRODUCTION.json').read_text())['status']=='PASS'
    patient_frames=[]; channel_frames=[]; records=[]; initial={}
    for fold in range(1,6):
        d=torch.load(root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt',weights_only=False)
        ids=set(d['patient'][d['val']]); expected=set(zip(d['patient'][d['val']],d['channel'][d['val']]))
        for arm in ['A0','B1','B2']:
            cell=root/f'fold{fold}'/arm
            meta=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            assert sha(cell/'LAST_PRIVATE.pt')==meta['last_sha256']
            assert sha(cell/'BEST_PRIVATE.pt')==meta['best_sha256']
            last=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            best=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert last['binding']==best['binding']==meta['binding']
            assert 'optimizer' in last and 'rng' in last and len(last['history'])==last['epoch']
            assert all(torch.isfinite(v).all() for v in last['model'].values())
            assert state_hash(last['best']['model'])==state_hash(best['model'])
            initial.setdefault(fold,meta['binding']['initial_hash'])
            assert initial[fold]==meta['binding']['initial_hash']
            c=pd.read_csv(cell/'VALIDATION_CHANNEL_PRIVATE.csv')
            assert set(zip(c.patient,c.channel))==expected and len(c)==len(expected)
            assert np.isfinite(c.score_nez).all() and ((c.score_nez>=0)&(c.score_nez<=1)).all()
            assert c.threshold.nunique()==1 and c.threshold.iloc[0]==meta['threshold']
            old=pd.read_csv(cell/'VALIDATION_PATIENT_PRIVATE.csv')
            # Recompute metrics from the frozen prediction file; no new inference.
            new=patient_metrics(c.y_nez.to_numpy(),c.score_nez.to_numpy(),c.patient.to_numpy(),c.center.to_numpy(),meta['threshold'])
            pair=new.merge(old,on=['patient','center'],validate='one_to_one',suffixes=('_n','_o'))
            assert set(new.patient)==ids
            for metric in METRICS: assert np.allclose(pair[metric+'_n'],pair[metric+'_o'],atol=1e-12,rtol=0)
            new['arm']=arm; new['fold']=fold
            patient_frames.append(new); channel_frames.append(c)
            records.append({'fold':fold,'arm':arm,'epochs':last['epoch'],
                'selected_epoch':meta['selected_epoch'],'threshold':meta['threshold'],
                'best_sha256':meta['best_sha256'],'last_sha256':meta['last_sha256'],
                'private_prediction_sha256':sha(cell/'VALIDATION_CHANNEL_PRIVATE.csv')})
    patients=pd.concat(patient_frames,ignore_index=True); channels=pd.concat(channel_frames,ignore_index=True)
    assert patients.groupby(['arm','fold']).size().eq(13).all()
    assert patients.patient.nunique()==47
    json_write(root/'public/ARTIFACT_INTEGRITY_AUDIT.json',{'status':'PASS','cells':records,
        'student_cells':15,'same_initial_state_within_fold':True,
        'optimizer_rng_history_hash_and_prediction_metrics_verified':True,
        'outer_evaluation_run':False})
    return patients,channels


def decisions(patients, channels, output):
    rows=[]; fixed=[]; descriptive=[]; shifts=[]
    baseline=channels[channels.arm=='A0']
    for arm in ['B1','B2']:
        pair=baseline.merge(channels[channels.arm==arm],on=['patient','channel','center','fold','y_nez'],
            validate='one_to_one',suffixes=('_a0','_method'))
        a0ok=(pair.score_nez_a0>=pair.threshold_a0)==pair.y_nez
        for operating in ['each_validation_selected','frozen_A0_numeric']:
            tau=pair.threshold_method if operating=='each_validation_selected' else pair.threshold_a0
            newok=(pair.score_nez_method>=tau)==pair.y_nez
            pair['corrected']=(~a0ok)&newok; pair['spoiled']=a0ok&~newok
            for level in ['all','fold','center']:
                pieces=[('all',pair)] if level=='all' else pair.groupby(level)
                for group,g in pieces:
                    rows.append({'arm':arm,'operating_point':operating,'level':level,'group':str(group),
                        'channels':len(g),'corrected_A0_errors':int(g.corrected.sum()),
                        'spoiled_A0_correct':int(g.spoiled.sum()),
                        'net_corrected':int(g.corrected.sum()-g.spoiled.sum()),
                        'mean_score_shift_nez':float((g.score_nez_method-g.score_nez_a0).mean()),
                        'mean_absolute_score_change':float(abs(g.score_nez_method-g.score_nez_a0).mean())})
        for fold,g in pair.groupby('fold'):
            tau=float(g.threshold_a0.iloc[0])
            m=patient_metrics(g.y_nez.to_numpy(),g.score_nez_method.to_numpy(),g.patient.to_numpy(),g.center.to_numpy(),tau)
            m['arm']=arm; m['fold']=fold; fixed.append(m)
            for pid,h in g.groupby('patient'):
                shifts.append({'arm':arm,'fold':fold,'patient':pid,
                    'signed_score_shift_nez':float((h.score_nez_method-h.score_nez_a0).mean()),
                    'absolute_score_change':float(abs(h.score_nez_method-h.score_nez_a0).mean())})
        pat=patients[patients.arm=='A0'].merge(patients[patients.arm==arm],
            on=['patient','center','fold'],suffixes=('_a0','_method'),validate='one_to_one')
        pat['A0_performance_bin']=pd.cut(pat.macro_f1_a0,[0,.5,.65,1],include_lowest=True,
            labels=['A0_F1_le_0_5','A0_F1_0_5_to_0_65','A0_F1_gt_0_65'])
        for name,g in pat.groupby('A0_performance_bin',observed=True):
            descriptive.append({'arm':arm,'A0_performance_bin':str(name),'patient_fold_cells':len(g),
                'unique_patients':g.patient.nunique(),'A0_macro_f1':float(g.macro_f1_a0.mean()),
                'macro_f1_delta':float((g.macro_f1_method-g.macro_f1_a0).mean()),
                'ez_auprc_delta':float((g.ez_auprc_method-g.ez_auprc_a0).mean()),
                'ez_auroc_delta':float((g.ez_auroc_method-g.ez_auroc_a0).mean()),
                'descriptive_not_selection':True})
    pd.DataFrame(rows).to_csv(output/'DECISION_CHANGE_AUDIT.csv',index=False)
    pd.DataFrame(descriptive).to_csv(output/'A0_POOR_PATIENT_DIAGNOSTIC.csv',index=False)
    fixed=pd.concat(fixed+[patients[patients.arm=='A0']],ignore_index=True)
    fixed_summary=aggregate(fixed,['arm']); fixed_summary.to_csv(output/'FIXED_A0_THRESHOLD_DIAGNOSTIC.csv',index=False)
    pd.DataFrame(shifts).groupby('arm')[['signed_score_shift_nez','absolute_score_change']].mean().reset_index().to_csv(output/'SCORE_SHIFT_DIAGNOSTIC.csv',index=False)
    return fixed_summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True); a=parser.parse_args()
    torch.set_num_threads(2); output=a.runtime/'public'
    patients,channels=verify(a.runtime,a.protocol)
    byfold=aggregate(patients,['arm','fold']); bycenter=aggregate(patients,['arm','center'])
    summary=aggregate(patients,['arm']); boot=bootstrap(patients)
    for name,frame in [('VALIDATION_BY_FOLD',byfold),('VALIDATION_BY_CENTER',bycenter),
        ('VALIDATION_SUMMARY',summary),('PAIRED_BOOTSTRAP',boot)]:frame.to_csv(output/(name+'.csv'),index=False)
    contrasts=[]
    for left,right in [('B1','A0'),('B2','A0'),('B2','B1')]:
        for fold in range(1,6):
            l=byfold[(byfold.arm==left)&(byfold.fold==fold)].iloc[0]
            r=byfold[(byfold.arm==right)&(byfold.fold==fold)].iloc[0]
            contrasts.append({'comparison':left+'-'+right,'fold':fold,**{m:float(l[m]-r[m]) for m in METRICS}})
    pd.DataFrame(contrasts).to_csv(output/'VALIDATION_FOLD_CONTRASTS.csv',index=False)
    fixed=decisions(patients,channels,output).set_index('arm')
    s=summary.set_index('arm'); delta=lambda m,l,r:float(s.loc[l,m]-s.loc[r,m])
    basecenter=bycenter[bycenter.arm=='A0'].set_index('center')
    targetcenter=bycenter[bycenter.arm=='B2'].set_index('center')
    worst_delta=float(targetcenter.macro_f1.min()-basecenter.macro_f1.min())
    worstname=basecenter.macro_f1.idxmin()
    improving=sum(r['macro_f1']>0 for r in contrasts if r['comparison']=='B2-A0')
    conditions={'B2_A0_macro_ge_0_020':delta('macro_f1','B2','A0')>=.02,
        'B2_B1_macro_ge_0_010':delta('macro_f1','B2','B1')>=.01,
        'EZ_AP_nondecline':delta('ez_auprc','B2','A0')>=0,
        'EZ_F1_nondecline':delta('ez_f1','B2','A0')>=0,
        'four_of_five_folds_improve':improving>=4,'worst_center_nondecline':worst_delta>=0}
    passed=all(conditions.values()); target=delta('macro_f1','B2','A0')>=.03
    if passed and target:terminal='PLLR_TARGET_GAIN_SUPPORTED'
    elif delta('macro_f1','B2','B1')<=0:terminal='PLLR_NO_INCREMENT_OVER_PERMUTATION'
    elif delta('macro_f1','B2','A0')<=0:terminal='PLLR_NOT_SUPPORTED'
    else:terminal='PLLR_POSITIVE_BELOW_TARGET'
    gate={'pass':passed,'conditions':conditions,'failed_conditions':[k for k,v in conditions.items() if not v],
        'terminal':terminal,'target_gain_ge_0_030':target,'improved_folds':improving,
        'B2_A0_macro_delta':delta('macro_f1','B2','A0'),'B2_B1_macro_delta':delta('macro_f1','B2','B1'),
        'B2_A0_EZ_AP_delta':delta('ez_auprc','B2','A0'),'B2_A0_EZ_F1_delta':delta('ez_f1','B2','A0'),
        'worst_center_delta':worst_delta,'A0_worst_center':worstname,
        'B2_change_at_A0_worst_center':float(targetcenter.loc[worstname,'macro_f1']-basecenter.loc[worstname,'macro_f1']),
        'diagnostic_fixed_A0_threshold_F1_delta':float(fixed.loc['B2','macro_f1']-fixed.loc['A0','macro_f1']),
        'ranking_gain_not_implied_by_F1':True,'no_new_outer_evaluation_authorized':True}
    json_write(output/'VALIDATION_GATE.json',gate)
    json_write(output/'RUN_STATUS.json',{'status':'COMPLETE_DEVELOPMENT_'+('GATE_PASS' if passed else 'GATE_FAILED'),
        'terminal':terminal,'student_cells':15,'development_complete':True,'outer_evaluation_run':False,
        'validation_appearances_per_arm':65,'unique_validation_patients':47,
        'bootstrap_draws':10000,'protocol_sha256':sha(a.protocol),'finalizer_sha256':sha(Path(__file__))})
    print(json.dumps({'summary':summary.to_dict(orient='records'),'gate':gate},indent=2),flush=True)
    print('PLLR_AGGREGATION_COMPLETE',flush=True)


if __name__=='__main__':main()
